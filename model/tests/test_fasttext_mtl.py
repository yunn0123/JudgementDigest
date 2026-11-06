import unittest
import hashlib
import json
from pathlib import Path
import tempfile

import torch
from torch.nn import functional as F
from transformers import ElectraConfig, ElectraModel

from judgment_bert.fasttext_mtl import (FastTextElectraMTL, FastTextGradientMixer, clip_dense_and_sparse_gradients,
                                      masked_mtl_loss)


class TestFastTextMTL(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        encoder = ElectraModel(ElectraConfig(vocab_size=16, embedding_size=8, hidden_size=8,
            num_hidden_layers=2, num_attention_heads=2, intermediate_size=16,
            max_position_embeddings=16))
        self.model = FastTextElectraMTL(3, 4, buckets=16, dim=6, encoder=encoder,
                                       last_n=1, dropout=0, attention_dim=4)
        self.ids = torch.tensor([[2, 5, 3], [2, 6, 3]])
        self.bag = (torch.tensor([1, 2, 2, 3]), torch.tensor([0, 2]), torch.tensor([.6, .8, .8, .6]))

    def predict(self):
        return self.model(self.ids, self.ids.ne(0), *self.bag)

    def test_term_loss_updates_both_embeddings_and_last_electra_layer(self):
        out = self.predict()
        self.assertEqual(out['bin'].shape, (2, 7))
        self.assertEqual(out['reg'].shape, (2, 1))
        torch.testing.assert_close(out['attention'].sum(-1), torch.ones(2))
        loss = F.cross_entropy(out['bin'], torch.tensor([0, 3]))
        loss.backward()
        for branch in (self.model.accusation_fasttext, self.model.article_fasttext):
            self.assertTrue(branch.emb.weight.grad.is_sparse)
            self.assertGreater(branch.emb.weight.grad.coalesce().values().abs().sum(), 0)
            self.assertIsNone(branch.head.weight.grad)
        self.assertTrue(all(p.grad is None for p in self.model.encoder.encoder.layer[0].parameters()))
        self.assertTrue(any(p.grad is not None for p in self.model.encoder.encoder.layer[1].parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.encoder.embeddings.parameters()))

    def test_auxiliary_loss_updates_own_branch_without_updating_electra(self):
        out = self.predict()
        F.binary_cross_entropy_with_logits(out['accusation'], torch.ones(2, 3)).backward()
        self.assertIsNotNone(self.model.accusation_fasttext.emb.weight.grad)
        self.assertIsNotNone(self.model.accusation_fasttext.head.weight.grad)
        self.assertIsNone(self.model.article_fasttext.emb.weight.grad)
        self.assertTrue(all(p.grad is None for p in self.model.encoder.parameters()))

    def test_stop_gradient_blocks_term_loss_only(self):
        self.model.term_backprop = False
        out = self.predict()
        F.cross_entropy(out['bin'], torch.tensor([0, 3])).backward(retain_graph=True)
        for branch in (self.model.accusation_fasttext, self.model.article_fasttext):
            self.assertIsNone(branch.emb.weight.grad)
        self.assertIsNotNone(self.model.accusation_projection[0].weight.grad)
        F.binary_cross_entropy_with_logits(out['accusation'], torch.ones(2, 3)).backward()
        self.assertIsNotNone(self.model.accusation_fasttext.emb.weight.grad)

    def test_masked_losses_ignore_invalid_targets_and_optimizer_updates_both_parameter_types(self):
        out = self.predict()
        masks = dict(imprisonment=torch.tensor([True, False]), accusation=torch.tensor([True, False]),
                     article=torch.tensor([False, True]))
        targets = dict(bin=torch.tensor([1, 999]), log_months=torch.tensor([1., float('nan')]),
                       accusation=torch.tensor([[1., 0., 0.], [float('nan')]*3]),
                       article=torch.tensor([[float('nan')]*4, [0., 1., 0., 0.]]))
        weights = dict(accusation=torch.ones(3), article=torch.ones(4))
        loss, parts = masked_mtl_loss(out, targets, masks, weights)
        expected = (F.cross_entropy(out['bin'][:1], torch.tensor([1]))
                    + F.mse_loss(out['reg'][:1, 0], torch.tensor([1.])))
        torch.testing.assert_close(parts['term'], expected)
        sparse = self.model.sparse_parameters()
        sparse_ids = {id(p) for p in sparse}
        sparse_opt = torch.optim.SparseAdam(sparse, lr=.01)
        dense_opt = torch.optim.AdamW([p for p in self.model.parameters()
                                      if p.requires_grad and id(p) not in sparse_ids], lr=.001)
        before = [p.detach().clone() for p in sparse]
        head_before = self.model.head.bin.weight.detach().clone()
        loss.backward()
        norm = clip_dense_and_sparse_gradients(self.model.parameters(), max_norm=.1)
        self.assertTrue(torch.isfinite(norm))
        clipped = torch.stack([(p.grad.coalesce().values() if p.grad.is_sparse else p.grad).norm()
                               for p in self.model.parameters() if p.grad is not None]).norm()
        self.assertLessEqual(clipped.item(), .100001)
        sparse_opt.step(); dense_opt.step()
        for old, new in zip(before, sparse):
            self.assertFalse(torch.equal(old, new))
        self.assertFalse(torch.equal(head_before, self.model.head.bin.weight))

    def test_checkpoint_restores_all_three_tasks(self):
        self.model.eval()
        expected = self.predict()
        state = {k: v.clone() for k, v in self.model.state_dict().items()}
        with torch.no_grad():
            self.model.accusation_fasttext.emb.weight.add_(1)
        self.model.load_state_dict(state)
        for task, value in expected.items():
            torch.testing.assert_close(self.predict()[task], value)

    def test_gradient_mixing_after_accumulation_preserves_other_gradients(self):
        sparse = self.model.sparse_parameters()
        mixer = FastTextGradientMixer(sparse, .1)
        own_expected = [torch.zeros_like(p) for p in sparse]
        term_expected = [torch.zeros_like(p) for p in sparse]
        # Simulate two microbatches and AMP scaling; duplicate bucket indices must coalesce.
        scale = 128.
        for microbatch in range(2):
            out = self.predict()
            term = F.cross_entropy(out['bin'], torch.tensor([microbatch, 3]))
            own = .1 * sum(F.binary_cross_entropy_with_logits(out[t], torch.ones_like(out[t]))
                           for t in ('accusation', 'article'))
            for expected, loss in ((own_expected, own), (term_expected, term)):
                for i, g in enumerate(torch.autograd.grad(loss / 2, sparse, retain_graph=True)):
                    expected[i] += g.to_dense()
            mixer.accumulate(scale * own / 2, scale * term / 2)
            (scale * (own + term) / 2).backward()
        dense_before = {n:p.grad.clone() for n,p in self.model.named_parameters()
                        if p.grad is not None and not p.grad.is_sparse}
        stats = mixer.apply(scale)
        for p, own, term, stat in zip(sparse, own_expected, term_expected, stats):
            expected = own + .1 * own.norm() / term.norm() * term
            torch.testing.assert_close(p.grad.to_dense() / scale, expected)
            self.assertAlmostEqual(stat['ratio'], .1, places=6)
        for n,p in self.model.named_parameters():
            if n in dense_before:
                torch.testing.assert_close(p.grad, dense_before[n])
        self.assertIsNone(mixer.own)
        self.assertIsNone(mixer.term)

    def test_gradient_mixing_zero_and_nonfinite_gradients(self):
        p = torch.nn.Parameter(torch.zeros(4, 2))
        def sparse(values):
            return torch.sparse_coo_tensor(torch.tensor([[1]]), torch.tensor([values]), (4,2)).coalesce()
        for own, term, expected in (([0.,0.], [3.,4.], [0.,0.]),
                                     ([3.,4.], [0.,0.], [3.,4.])):
            mixer = FastTextGradientMixer([p], .1)
            mixer.own, mixer.term = [sparse(own)], [sparse(term)]
            stat = mixer.apply()[0]
            torch.testing.assert_close(p.grad.to_dense()[1], torch.tensor(expected))
            self.assertEqual(stat['ratio'], 0.)
        mixer.own, mixer.term = [sparse([1.,2.])], [sparse([float('inf'),0.])]
        self.assertFalse(mixer.apply()[0]['finite'])
        self.assertFalse(torch.isfinite(p.grad.coalesce().values()).all())

    def test_original_fasttext_checkpoints_load_and_class_order_is_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'fold_01').mkdir()
            classes = dict(accusation=['a', 'b', 'c'], article=['d', 'e', 'f', 'g'])
            manifest = dict(sha256={})
            expected = {}
            for task in classes:
                branch = getattr(self.model, task + '_fasttext')
                state = {'emb.weight': torch.randn_like(branch.emb.weight),
                         'heads.out.weight': torch.randn_like(branch.head.weight),
                         'heads.out.bias': torch.randn_like(branch.head.bias)}
                expected[task] = state
                torch.save(state, root / 'fold_01' / (task + '.pt'))
                metadata = dict(fold='fold_01', task=task, classes=classes[task], feature_dim=6,
                    article_field='relevant_articles_with_suffix', selected_epoch=3,
                    vectorizer=dict(n_features=16, analyzer='char', ngram_range=[1,2],
                                    norm='l2', alternate_sign=False))
                (root / 'fold_01' / (task + '.json')).write_text(json.dumps(metadata))
                for suffix in ('.pt', '.json'):
                    relative = 'fold_01/' + task + suffix
                    manifest['sha256'][relative] = hashlib.sha256((root/relative).read_bytes()).hexdigest()
            (root / 'fasttext_checkpoint_manifest.json').write_text(json.dumps(manifest))
            self.model.load_fasttext_checkpoints(root, 'fold_01', classes)
            for task in classes:
                branch = getattr(self.model, task + '_fasttext')
                torch.testing.assert_close(branch.emb.weight, expected[task]['emb.weight'])
                torch.testing.assert_close(branch.head.weight, expected[task]['heads.out.weight'])
            with self.assertRaisesRegex(ValueError, 'class order'):
                self.model.load_fasttext_checkpoints(root, 'fold_01', dict(classes, accusation=['b','a','c']))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
