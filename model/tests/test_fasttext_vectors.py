import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from judgment_bert.encoding import Collator
from judgment_bert.fasttext_vectors import FastTextVectorBank


class TestFastTextVectorBank(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'fold_01').mkdir()
        self.records = [dict(source_line=i + 1, fact=f'案件{i}', jid=f'id{i}') for i in range(3)]
        self.write()

    def write(self, article_field='relevant_articles_with_suffix', article_fold='fold_01'):
        for task, dim in (('accusation', 4), ('article', 6)):
            vectors = np.arange(3 * dim, dtype=np.float32).reshape(3, dim)
            metadata = dict(format_version=1, task=task, feature_dim=dim,
                            fold=article_fold if task == 'article' else 'fold_01',
                            article_field=article_field, train_source_lines=[1],
                            vectors_sha256=hashlib.sha256(vectors.tobytes()).hexdigest())
            np.savez(self.root / 'fold_01' / f'{task}.npz',
                     vectors=vectors,
                     source_lines=np.arange(1, 4), jids=np.array(['id0', 'id1', 'id2']),
                     fact_sha256=np.array([hashlib.sha256(r['fact'].encode()).hexdigest() for r in self.records]),
                     roles=np.array(['train', 'valid', 'test']), metadata=np.array(json.dumps(metadata)))

    def test_lookup_by_source_line_not_batch_position(self):
        bank = FastTextVectorBank(self.root, 'fold_01')
        result = bank.batch([self.records[2]], 'test')
        torch.testing.assert_close(result['accusation_vectors'], torch.tensor([[8., 9., 10., 11.]]))
        self.assertEqual(result['article_vectors'].shape, (1, 6))
        self.assertFalse(result['article_vectors'].requires_grad)

    def test_rejects_wrong_fact_role_missing_line_and_jid(self):
        bank = FastTextVectorBank(self.root, 'fold_01')
        for row, role in ((dict(self.records[0], fact='不同案件'), 'train'),
                          (self.records[0], 'test'),
                          (dict(self.records[0], source_line=99), 'train'),
                          (dict(self.records[0], jid='別案'), 'train')):
            with self.assertRaises(ValueError):
                bank.batch([row], role)

    def test_rejects_coarse_articles_and_wrong_fold(self):
        self.write(article_field='relevant_articles')
        with self.assertRaisesRegex(ValueError, 'suffix'):
            FastTextVectorBank(self.root, 'fold_01')
        self.write(article_fold='fold_02')
        with self.assertRaisesRegex(ValueError, 'fold'):
            FastTextVectorBank(self.root, 'fold_01')

    def test_collator_attaches_features_and_keeps_target(self):
        bank = FastTextVectorBank(self.root, 'fold_01')
        collate = bank.collator([self.records[1]], 'valid', Collator(0))
        batch, targets, indices = collate([dict(chunks=[[2, 3]], target=torch.tensor(1), index=0)])
        self.assertEqual(indices, [0])
        self.assertEqual(targets.tolist(), [1])
        self.assertEqual(batch['input_ids'].tolist(), [[2, 3]])
        torch.testing.assert_close(batch['accusation_vectors'], torch.tensor([[4., 5., 6., 7.]]))

    def test_training_entry_uses_exported_vectors_without_network(self):
        from transformers import BertTokenizerFast
        from judgment_bert.train import main
        prepared = self.root / 'prepared'
        prepared.mkdir()
        (prepared / 'manifest.json').write_text(json.dumps(dict(
            task='imprisonment', kind='multiclass', classes=['短刑', '長刑'], num_labels=2)))
        for i, role in enumerate(('train', 'valid', 'test')):
            (prepared / f'{role}.jsonl').write_text(json.dumps(
                dict(self.records[i], target=i % 2, true_labels=i % 2)) + '\n')
        vocab = self.root / 'vocab.txt'
        vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n案\n件\n0\n1\n2\n')
        tokenizer = BertTokenizerFast(vocab_file=str(vocab))
        output = self.root / 'term_run'
        with patch('sys.argv', ['train', '--data-dir', str(prepared), '--output', str(output),
                               '--fasttext-vectors', str(self.root), '--fasttext-fold', 'fold_01',
                               '--tiny-smoke', '--evaluate-test', '--device', 'cpu']), \
             patch('judgment_bert.train.AutoTokenizer.from_pretrained', return_value=tokenizer):
            main()
        state = torch.load(output / 'best.pt', weights_only=True)
        self.assertIn('accusation_projection.0.weight', state)
        self.assertIn('query.weight', state)
        self.assertAlmostEqual(state['beta'].item(), 0.1)
        self.assertTrue((output / 'fasttext_vector_metadata.json').exists())
        metrics = json.loads((output / 'metrics.json').read_text())
        self.assertTrue(metrics['tiny_random_smoke'])
        self.assertEqual(metrics['test_status'], 'evaluated')
