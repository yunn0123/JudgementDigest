import unittest

import torch
from transformers import ElectraConfig, ElectraModel

from judgment_bert.term_fusion import FrozenFastTextTermModel, FrozenFastTextDualTermModel


class TestTermFusion(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        encoder = ElectraModel(ElectraConfig(
            vocab_size=32, embedding_size=16, hidden_size=16, num_hidden_layers=2,
            num_attention_heads=2, intermediate_size=32, max_position_embeddings=16,
        ))
        self.model = FrozenFastTextTermModel(
            num_labels=3, accusation_dim=6, article_dim=8, attention_dim=4,
            encoder=encoder, last_n=1, dropout=0,
        )
        self.ids = torch.tensor([[2, 5, 3], [2, 6, 3]])
        self.accusation = torch.randn(2, 6, requires_grad=True)
        self.article = torch.randn(2, 8, requires_grad=True)

    def predict(self, **kwargs):
        return self.model(self.ids, self.ids.ne(0), self.accusation, self.article, **kwargs)

    def test_term_loss_trains_projections_attention_and_last_encoder_layer_only(self):
        logits, weights = self.predict(return_attention=True)
        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(weights.shape, (2, 2))
        torch.testing.assert_close(weights.sum(-1), torch.ones(2))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2])).backward()
        self.assertIsNone(self.accusation.grad)
        self.assertIsNone(self.article.grad)
        for module in (self.model.accusation_projection, self.model.article_projection,
                       self.model.query, self.model.key, self.model.value, self.model.head):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in module.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.encoder.embeddings.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.encoder.encoder.layer[0].parameters()))
        self.assertTrue(any(p.grad is not None for p in self.model.encoder.encoder.layer[1].parameters()))
        self.assertFalse(self.model.beta.requires_grad)

    def test_beta_zero_matches_encoder_only_and_ignores_auxiliary_changes(self):
        self.model.eval()
        self.model.beta.zero_()
        expected = self.model.head(self.model.encode(self.ids, self.ids.ne(0)))
        torch.testing.assert_close(self.predict(), expected)
        with torch.no_grad():
            self.accusation.add_(10)
            self.article.mul_(10)
        torch.testing.assert_close(self.predict(), expected)

    def test_positive_beta_uses_auxiliary_vectors(self):
        self.model.eval()
        before = self.predict()
        with torch.no_grad():
            self.accusation.add_(10)
        self.assertFalse(torch.allclose(before, self.predict()))

    def test_chunk_padding_does_not_change_prediction(self):
        self.model.eval()
        ids = torch.stack((self.ids, torch.zeros_like(self.ids)), dim=1)
        chunks = torch.tensor([[True, False], [True, False]])
        result = self.model(ids, ids.ne(0), self.accusation, self.article, chunk_mask=chunks)
        torch.testing.assert_close(result, self.predict())

    def test_rejects_bad_feature_shape_nonfinite_values_and_beta(self):
        with self.assertRaisesRegex(ValueError, 'accusation_vectors'):
            self.model(self.ids, self.ids.ne(0), torch.zeros(2, 7), self.article)
        with torch.no_grad():
            self.article[0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.predict()
        for beta in (-1, float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'beta'):
                FrozenFastTextTermModel(3, 6, 8, beta=beta)

    def test_learnable_beta_updates_and_checkpoint_restores_effective_value(self):
        model = FrozenFastTextTermModel(3, 6, 8, beta=0.2, learnable_beta=True,
                                       encoder=self.model.encoder, last_n=1, dropout=0)
        self.assertAlmostEqual(model.effective_beta().item(), 0.2, places=6)
        optimizer = torch.optim.SGD([model.beta_logit], lr=0.5)
        logits = model(self.ids, self.ids.ne(0), self.accusation, self.article)
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2])).backward()
        self.assertIsNotNone(model.beta_logit.grad)
        self.assertGreater(abs(model.beta_logit.grad.item()), 0)
        optimizer.step()
        learned = model.effective_beta().item()
        self.assertNotEqual(learned, 0.2)
        self.assertTrue(0 < learned < 1)
        restored = FrozenFastTextTermModel(3, 6, 8, beta=0.2, learnable_beta=True,
                                          encoder=self.model.encoder, last_n=1, dropout=0)
        restored.load_state_dict(model.state_dict())
        self.assertEqual(restored.effective_beta().item(), learned)
        self.assertIsNone(self.accusation.grad)
        self.assertIsNone(self.article.grad)

    def test_dual_term_head_matches_baseline_outputs_and_loss(self):
        model = FrozenFastTextDualTermModel(6, 8, beta=0.2, learnable_beta=True,
                                            encoder=self.model.encoder, last_n=1, dropout=0)
        output = model(self.ids, self.ids.ne(0), self.accusation, self.article)
        self.assertEqual(output['bin'].shape, (2, 7))
        self.assertEqual(output['reg'].shape, (2, 1))
        loss = (torch.nn.functional.cross_entropy(output['bin'], torch.tensor([1, 2]))
                + torch.nn.functional.mse_loss(output['reg'][:, 0], torch.log1p(torch.tensor([6., 12.]))))
        loss.backward()
        self.assertIsNotNone(model.beta_logit.grad)
        self.assertIsNone(self.accusation.grad)
        self.assertIsNone(self.article.grad)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
