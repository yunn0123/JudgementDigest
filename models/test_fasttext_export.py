import json
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from sklearn.feature_extraction.text import HashingVectorizer

from fasttext_export import export_vectors


class TestFastTextExport(unittest.TestCase):
    def test_real_training_script_exports_both_suffix_task_vectors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            roles = ['train'] * 4 + ['valid'] * 2 + ['test'] * 2 + ['holdout'] * 2 + ['purged']
            labels = ['刑法第185條之3', '刑法第339條之4']
            rows = [dict(fact=f'案件{i}：' + ('駕車飲酒' if i % 2 == 0 else '詐取財物'),
                         meta=dict(jid=f'case{i}', year=2025 if role == 'holdout' else 2024,
                                   accusation=['罪甲' if i % 2 == 0 else '罪乙'],
                                   relevant_articles=['粗條'],
                                   relevant_articles_with_suffix=['門檻外條號' if i == 2 else labels[i % 2]],
                                   task_validity=dict(accusation=True, article=True, imprisonment=True),
                                   term_of_imprisonment=dict(imprisonment=6, death_penalty=False, life_imprisonment=False)))
                    for i, role in enumerate(roles)]
            (root / 'shared_100k_with_article_suffix.jsonl').write_text(
                ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
            (root / 'fixed_classes_coarse.json').write_text(json.dumps(
                dict(accusation=['罪甲', '罪乙'], article=['粗條'])), encoding='utf-8')
            (root / 'fixed_classes_suffix.json').write_text(json.dumps(dict(article=labels)), encoding='utf-8')
            with (root / 'shared_fold_assignments.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=['sample_line_1based', 'jid', 'fold_01'])
                writer.writeheader()
                writer.writerows(dict(sample_line_1based=i + 1, jid=f'case{i}', fold_01=role)
                                 for i, role in enumerate(roles))
            result = subprocess.run([
                sys.executable, str(Path(__file__).with_name('fasttext_gpu.py')),
                '--shared', str(root), '--article-field', 'relevant_articles_with_suffix',
                '--tasks', 'accusation,article', '--folds', 'fold_01', '--epochs', '1',
                '--dim', '4', '--buckets', '32', '--batch', '4',
                '--save-checkpoints',
                '--out', str(root / 'smoke.md'), '--export-vectors', str(root / 'vectors'),
            ], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for task in ('accusation', 'article'):
                state = torch.load(root / 'vectors' / 'fold_01' / f'{task}.pt', weights_only=True)
                self.assertEqual(state['emb.weight'].shape, (32, 4))
                with np.load(root / 'vectors' / 'fold_01' / f'{task}.npz', allow_pickle=False) as archive:
                    info = json.loads(archive['metadata'].item())
                    self.assertEqual(archive['vectors'].shape, (11, 4))
                    self.assertEqual(archive['roles'].tolist(), roles)
                    self.assertEqual(info['selected_epoch'], 1)
                    self.assertEqual(set(info['train_source_lines']), {1, 2, 4} if task == 'article' else {1, 2, 3, 4})
                    if task == 'article':
                        self.assertEqual(info['classes'], labels)

    def test_exports_prehead_vectors_in_eval_and_never_overwrites(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.EmbeddingBag(32, 4, mode='sum', sparse=True)
                self.drop = torch.nn.Dropout(0.2)

        model = Model()
        hv = HashingVectorizer(analyzer='char', ngram_range=(1, 2), n_features=32,
                               alternate_sign=False, norm='l2', dtype=np.float32)
        rows = [dict(jid='a', fact='甲案件'), dict(jid='b', fact='乙案件')]
        assignments = [dict(sample_line_1based=str(i + 1), jid=row['jid'], fold_01=role)
                       for i, (row, role) in enumerate(zip(rows, ['train', 'test']))]
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'accusation.npz'
            metadata = dict(task='accusation', article_field='relevant_articles_with_suffix')
            export_vectors(model, hv, rows, assignments, 'fold_01', rows[:1], output, metadata, batch_size=1)
            self.assertFalse(model.training)
            matrix = hv.transform([r['fact'] for r in rows]).tocsr()
            matrix.sort_indices()
            expected = model.emb(torch.from_numpy(matrix.indices.astype(np.int64)),
                                 torch.from_numpy(matrix.indptr[:-1].astype(np.int64)),
                                 per_sample_weights=torch.from_numpy(matrix.data)).detach().numpy()
            with np.load(output, allow_pickle=False) as data:
                np.testing.assert_allclose(data['vectors'], expected)
                self.assertEqual(data['source_lines'].tolist(), [1, 2])
                self.assertEqual(data['roles'].tolist(), ['train', 'test'])
                self.assertEqual(json.loads(data['metadata'].item())['train_source_lines'], [1])
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            with self.assertRaises(FileExistsError):
                export_vectors(model, hv, rows, assignments, 'fold_01', rows[:1], output, metadata)
            with self.assertRaisesRegex(ValueError, 'TRAIN'):
                export_vectors(model, hv, rows, assignments, 'fold_01', rows[1:], Path(temp) / 'bad.npz', metadata)
