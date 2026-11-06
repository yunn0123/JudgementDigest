"""Export frozen, per-document EmbeddingBag features for term ELECTRA."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def export_vectors(model, vectorizer, rows, assignments, fold, train_rows,
                   output, metadata, batch_size=4096):
    """Export all assigned documents in source-line order, including task-invalid rows."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not rows or batch_size < 1:
        raise ValueError('Need documents and a positive export batch size')
    roles = {int(a['sample_line_1based']): a[fold] for a in assignments}
    if sorted(roles) != list(range(1, len(rows) + 1)) or len(assignments) != len(rows):
        raise ValueError('Assignments must cover source lines exactly once')
    for a in assignments:
        if rows[int(a['sample_line_1based']) - 1]['jid'] != a['jid']:
            raise ValueError('Assignment jid mismatch')
    row_ids = {id(row): i + 1 for i, row in enumerate(rows)}
    train_lines = [row_ids[id(row)] for row in train_rows]
    if any(roles[line] != 'train' for line in train_lines):
        raise ValueError('fastText training must use TRAIN rows only')
    device = model.emb.weight.device
    vectors = np.empty((len(rows), model.emb.embedding_dim), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            part = rows[start:start + batch_size]
            matrix = vectorizer.transform([r['fact'] for r in part]).tocsr()
            matrix.sort_indices()
            indices = torch.from_numpy(matrix.indices.astype(np.int64)).to(device)
            offsets = torch.from_numpy(matrix.indptr[:-1].astype(np.int64)).to(device)
            weights = torch.from_numpy(matrix.data.astype(np.float32)).to(device)
            features = model.emb(indices, offsets, per_sample_weights=weights)
            vectors[start:start + len(part)] = features.float().cpu().numpy()
    if not np.isfinite(vectors).all():
        raise ValueError('Nonfinite exported vectors')
    info = dict(metadata, format_version=1, fold=fold, feature_dim=vectors.shape[1],
                vectors_sha256=hashlib.sha256(vectors.tobytes()).hexdigest(),
                feature='EmbeddingBag before classification head; eval; no dropout',
                hash_normalization='fact.strip()', train_source_lines=train_lines,
                vectorizer={k: v for k, v in vectorizer.get_params().items()
                            if k in ('analyzer', 'ngram_range', 'n_features', 'alternate_sign', 'norm', 'lowercase')})
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting a previous fold/task export.
    with output.open('xb') as handle:
        np.savez(handle, vectors=vectors, source_lines=np.arange(1, len(rows) + 1),
                 jids=np.asarray([r['jid'] for r in rows], dtype=str),
                 fact_sha256=np.asarray([hashlib.sha256(r['fact'].strip().encode()).hexdigest() for r in rows]),
                 roles=np.asarray([roles[i] for i in range(1, len(rows) + 1)]),
                 metadata=np.asarray(json.dumps(info, ensure_ascii=False)))
