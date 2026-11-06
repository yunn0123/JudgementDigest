"""Load per-fold fastText document vectors, with source/role alignment checks."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


class FastTextVectorBank:
    def __init__(self, folder, fold):
        self.fold = fold
        self.data = {}
        self.metadata = {}
        for task in ('accusation', 'article'):
            with np.load(Path(folder) / fold / f'{task}.npz', allow_pickle=False) as archive:
                info = json.loads(str(archive['metadata'].item()))
                if info.get('format_version') != 1 or info['fold'] != fold or info['task'] != task:
                    raise ValueError('Vector format, task or fold mismatch')
                if task == 'article' and info['article_field'] != 'relevant_articles_with_suffix':
                    raise ValueError('Expected suffix-preserving article vectors')
                vectors = archive['vectors'].copy()
                lines, hashes, roles, jids = (archive[k].copy() for k in
                                              ('source_lines', 'fact_sha256', 'roles', 'jids'))
            n = len(lines)
            if (vectors.shape != (n, info['feature_dim']) or not np.isfinite(vectors).all()
                    or not np.issubdtype(vectors.dtype, np.floating)
                    or any(a.shape != (n,) for a in (lines, hashes, roles, jids))
                    or len(set(lines.tolist())) != n):
                raise ValueError('Invalid vector dimensions, values or source-line keys')
            if hashlib.sha256(vectors.tobytes()).hexdigest() != info['vectors_sha256']:
                raise ValueError('Vector checksum mismatch')
            index = {int(line): i for i, line in enumerate(lines)}
            if any(line not in index or roles[index[line]] != 'train'
                   for line in info['train_source_lines']):
                raise ValueError('Vector metadata reports training outside TRAIN')
            self.data[task] = (vectors, index, hashes, roles, jids)
            self.metadata[task] = info
        left, right = self.data['accusation'], self.data['article']
        if left[1] != right[1] or any(not np.array_equal(left[i], right[i]) for i in (2, 3, 4)):
            raise ValueError('Accusation/article documents or split assignments differ')

    def batch(self, records, role):
        """records use prepared ELECTRA source_line and fact, never row position."""
        if role not in ('train', 'valid', 'test', 'holdout') or not records:
            raise ValueError('Need a nonempty batch and explicit shared split role')
        result = {}
        for task, (vectors, index, hashes, roles, jids) in self.data.items():
            selected = []
            for row in records:
                line = row['source_line']
                if line not in index:
                    raise ValueError(f'Missing {task} vector at source line {line}')
                i = index[line]
                digest = hashlib.sha256(row['fact'].strip().encode()).hexdigest()
                if hashes[i] != digest or roles[i] != role:
                    raise ValueError(f'Fact or split mismatch at source line {line}')
                if row.get('jid') is not None and row['jid'] != jids[i]:
                    raise ValueError(f'jid mismatch at source line {line}')
                selected.append(i)
            result[f'{task}_vectors'] = torch.from_numpy(vectors[selected].copy()).float()
        return result

    def collator(self, records, role, base_collator):
        """Wrap the existing token collator; shuffled dataset indexes stay aligned."""
        # Validate the complete split before starting an epoch.
        self.batch(records, role)

        def collate(items):
            batch, target, indices = base_collator(items)
            batch.update(self.batch([records[i] for i in indices], role))
            return batch, target, indices

        return collate
