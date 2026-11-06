"""Fold-local OOF fastText probabilities, with fixed class-coordinate features.

Inner models start from scratch and exclude their OOF rows from fitting and
selection. The original outer VALID selects epochs, as in the original fastText
run. Outer-full checkpoints supply VALID/TEST/2025 features. No ELECTRA training.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.special import expit
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch import nn
from torch.nn import functional as F
import evaluate as E

TASKS = ('accusation', 'article')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def grouped_inner_folds(rows, train, count, seed):
    """Keep identical stripped facts and identical case IDs in the same inner fold."""
    parent = {int(i): int(i) for i in train}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i in train:
        i = int(i)
        keys = [('fact', hashlib.sha256(rows[i]['fact'].strip().encode()).hexdigest()),
                ('jid', rows[i]['jid'])]
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = {}
    for i in train:
        groups.setdefault(find(int(i)), []).append(int(i))
    groups = list(groups.values())
    if len(groups) < count:
        raise ValueError('Too few independent TRAIN groups for inner folds')
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    sizes = np.zeros(count, dtype=int)
    assigned = np.full(len(rows), -1, dtype=np.int16)
    for group in groups:
        k = int(sizes.argmin())
        assigned[group] = k
        sizes[k] += len(group)
    assert np.all(assigned[train] >= 0) and np.all(sizes > 0)
    return assigned


class FastText(nn.Module):
    def __init__(self, buckets, dim, labels):
        super().__init__()
        self.emb = nn.EmbeddingBag(buckets, dim, mode='sum', sparse=True)
        nn.init.normal_(self.emb.weight, std=.05)
        self.drop = nn.Dropout(.2)
        self.heads = nn.ModuleDict({'out': nn.Linear(dim, labels)})

    def forward(self, X, ids):
        sub = X[ids]
        h = self.emb(torch.from_numpy(sub.indices.astype(np.int64)),
                     torch.from_numpy(sub.indptr[:-1].astype(np.int64)),
                     per_sample_weights=torch.from_numpy(sub.data))
        return self.heads['out'](self.drop(h))


def predict(model, X, ids):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(X, ids[start:start+4096]).numpy()
                               for start in range(0, len(ids), 4096)])


def score(Y, logits):
    return {k: float(v) for k, v in E.multilabel_scores(Y, E.at_least_one(logits)).items()
            if np.isfinite(v)}


def fit_inner(args, X, Y, train, valid, dest, metadata):
    if not len(train) or not len(valid) or np.intersect1d(train, valid).size:
        raise ValueError('Need disjoint nonempty TRAIN and VALID')
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    model = FastText(args.buckets, args.dim, Y.shape[1])
    sparse = torch.optim.SparseAdam(model.emb.parameters(), lr=args.lr)
    dense = torch.optim.Adam(model.heads.parameters(), lr=args.lr)
    positive = Y[train].sum(0)
    pw = torch.from_numpy(np.clip(np.sqrt((len(train)-positive)/np.maximum(positive, 1)), 1, 30).astype(np.float32))
    target = torch.from_numpy(Y)
    best, best_epoch, best_state, stale = -1., 0, None, 0
    history, started = [], time.time()
    dest.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs+1):
        model.train()
        order = rng.permutation(train)
        total = 0.
        for start in range(0, len(order), args.batch):
            ids = order[start:start+args.batch]
            loss = F.binary_cross_entropy_with_logits(model(X, ids), target[ids], pos_weight=pw)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite fastText loss')
            sparse.zero_grad(set_to_none=True)
            dense.zero_grad(set_to_none=True)
            loss.backward()
            sparse.step()
            dense.step()
            total += loss.item()*len(ids)
        metrics = score(Y[valid], predict(model, X, valid))
        selection = (metrics['micro']+metrics['macro'])/2
        history.append(dict(epoch=epoch, loss=total/len(train), valid=metrics,
                            elapsed_seconds=time.time()-started))
        if selection > best:
            best, best_epoch, best_state, stale = selection, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        write(dest.with_suffix('.history.json'), history)
        print(f"{metadata['fold']} inner={metadata['inner_fold']} {metadata['task']} "
              f"epoch={epoch} loss={total/len(train):.5f} valid={selection:.5f} "
              f"seconds={time.time()-started:.0f}", flush=True)
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    tmp = dest.with_suffix('.pt.tmp')
    torch.save(model.state_dict(), tmp)
    tmp.replace(dest)
    info = dict(metadata, selected_epoch=best_epoch, valid_score=best,
                train_source_lines=(train+1).tolist(), selection_source_lines=(valid+1).tolist(),
                checkpoint_sha256=sha(dest), history=history)
    write(dest.with_suffix('.json'), info)
    return model, info


def export(path, vectors, rows, roles, metadata, producer):
    if not np.isfinite(vectors).all() or np.any(vectors < 0) or np.any(vectors > 1):
        raise ValueError('Invalid sigmoid feature values')
    info = dict(metadata, format_version=1, feature_dim=vectors.shape[1],
                feature='sigmoid classifier scores; fixed class order; not calibrated probabilities',
                vectors_sha256=hashlib.sha256(vectors.tobytes()).hexdigest())
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with path.open('xb') as f:
        np.savez_compressed(f, vectors=vectors, source_lines=np.arange(1,len(rows)+1),
                            jids=np.asarray([r['jid'] for r in rows]), roles=roles,
                            fact_sha256=np.asarray([hashlib.sha256(r['fact'].strip().encode()).hexdigest() for r in rows]),
                            producer_inner_fold=producer,
                            metadata=np.asarray(json.dumps(info, ensure_ascii=False)))


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    manifest = dict(status='running', args=vars(args), device='cpu',
                    representation='sigmoid scores: accusation 80 classes / article 109 classes; no hidden-vector mixing',
                    inner_selection='outer VALID only; OOF rows excluded from fitting and epoch selection',
                    inference='original corresponding outer-full fastText checkpoint; no inner ensemble',
                    control='same-class-coordinate in-sample features also exported for a fair OOF comparison',
                    completed_models=0, started_unix=time.time())
    write(out/'run_manifest.json', manifest)
    try:
        rows, assignments, classes = E.load_shared(args.shared, 'relevant_articles_with_suffix')
        assignments = sorted(assignments, key=lambda a: int(a['sample_line_1based']))
        all_ids = np.arange(len(rows))
        data_sha = {n:sha(Path(args.shared)/n) for n in ['shared_100k_with_article_suffix.jsonl','shared_fold_assignments.csv']}
        original_manifest = json.loads(Path(args.checkpoint_manifest).read_text())
        if data_sha != original_manifest['data_sha256']:
            raise ValueError('Original checkpoint data or folds differ')
        manifest['data_sha256'] = data_sha
        hv = HashingVectorizer(analyzer='char', ngram_range=(1,2), n_features=args.buckets,
                               alternate_sign=False, norm='l2', dtype=np.float32)
        print('Hashing all facts with stateless HashingVectorizer', flush=True)
        X = hv.transform([r['fact'] for r in rows]).tocsr()
        X.sort_indices()
        print(f'FEATURES READY rows={len(rows)} nnz={X.nnz}', flush=True)
        results = []
        for fold in E.shared_folds(assignments):
            roles = np.asarray([a[fold] for a in assignments])
            train_all = np.flatnonzero(roles=='train')
            inner = grouped_inner_folds(rows, train_all, args.inner_folds, args.seed)
            write(out/'assignments'/f'{fold}.json', dict(fold=fold, seed=args.seed,
                grouping='connected components of identical jid or stripped fact SHA256',
                train_source_lines=(train_all+1).tolist(), inner_fold_0based=inner[train_all].tolist()))
            for task in TASKS:
                valid_task = np.array([r['valid'][task] for r in rows])
                Y = MultiLabelBinarizer(classes=classes[task]).fit_transform(
                    [r[task] if valid_task[i] else [] for i,r in enumerate(rows)]).astype(np.float32)
                train = np.flatnonzero((roles=='train') & valid_task)
                valid = np.flatnonzero((roles=='valid') & valid_task)
                source = Path(args.original_vectors)/fold
                with np.load(source/(task+'.npz'), allow_pickle=False) as archive:
                    original = json.loads(archive['metadata'].item())
                    if (original['fold']!=fold or original['classes']!=classes[task]
                            or not np.array_equal(archive['roles'], roles)
                            or set(original['train_source_lines'])!=set((train+1).tolist())
                            or not np.array_equal(archive['source_lines'], all_ids+1)):
                        raise ValueError('Original vector metadata does not match current fold')
                checkpoint = source/(task+'.pt')
                if sha(checkpoint)!=original_manifest['sha256'][f'{fold}/{task}.pt']:
                    raise ValueError('Original checkpoint checksum mismatch')
                base_meta = dict(fold=fold,task=task,classes=classes[task],
                    article_field='relevant_articles_with_suffix', train_source_lines=(train+1).tolist(),
                    vectorizer=dict(analyzer='char',ngram_range=[1,2],n_features=args.buckets,alternate_sign=False,norm='l2'),
                    source_data_sha256=data_sha, original_checkpoint_sha256=sha(checkpoint))
                full_model = FastText(args.buckets,args.dim,len(classes[task]))
                full_model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True))
                full_logits = predict(full_model,X,all_ids)
                full = expit(full_logits).astype(np.float32)
                export(out/'in_sample_probability_control'/fold/(task+'.npz'),full,rows,roles,
                       dict(base_meta,train_features='in-sample; original full-TRAIN model'),np.full(len(rows),-1,dtype=np.int16))
                del full_model
                oof = full.copy()
                covered = np.zeros(len(rows),dtype=np.int8)
                inner_reports = []
                for k in range(args.inner_folds):
                    fit = train[inner[train]!=k]
                    held = train_all[inner[train_all]==k]
                    if np.intersect1d(fit,held).size or np.intersect1d(valid,held).size:
                        raise ValueError('OOF rows used for fitting or selection')
                    dest = out/'checkpoints'/fold/f'inner_{k+1:02}'/(task+'.pt')
                    model, info = fit_inner(args,X,Y,fit,valid,dest,
                        dict(fold=fold,inner_fold=k,task=task,classes=classes[task],
                             oof_source_lines=(held+1).tolist(),initialization='random from scratch',
                             selection='outer VALID micro/macro F1 mean',seed=args.seed))
                    logits = predict(model,X,held)
                    oof[held] = expit(logits)
                    covered[held] += 1
                    labelled = valid_task[held]
                    info['oof_metrics'] = score(Y[held[labelled]],logits[labelled])
                    write(dest.with_suffix('.json'),info)
                    inner_reports.append(dict(inner_fold=k,selected_epoch=info['selected_epoch'],
                                               n_train=len(fit),n_oof=len(held),oof_metrics=info['oof_metrics']))
                    del model
                    manifest['completed_models'] += 1
                    manifest['current'] = dict(fold=fold,task=task,inner_fold=k)
                    write(out/'run_manifest.json',manifest)
                if not np.all(covered[train_all]==1) or np.any(covered[roles!='train']):
                    raise ValueError('OOF coverage must be exactly once for TRAIN and zero elsewhere')
                if not np.array_equal(oof[roles!='train'],full[roles!='train']):
                    raise ValueError('Non-TRAIN features changed')
                export(out/'vectors'/fold/(task+'.npz'),oof,rows,roles,
                    dict(base_meta,train_features='5-inner-fold OOF sigmoid scores',inner_folds=args.inner_folds,
                         held_out_rows_used_for_selection=False,inner_models=inner_reports,
                         non_train_features='original full-TRAIN checkpoint'),inner)
                # sigmoid > .5 matches logits > 0 with the same at-least-one fallback.
                metrics = {}
                for role in ('train','valid','test','holdout'):
                    selected = np.flatnonzero((roles==role)&valid_task)
                    for cohort, ids in [('all_task_valid',selected),('imprisonment_only',np.array([i for i in selected if rows[i]['valid']['imprisonment']],dtype=int))]:
                        if len(ids):
                            metrics[f'{role}/{cohort}'] = dict(n=len(ids),
                                oof=score(Y[ids],oof[ids]-.5),in_sample_control=score(Y[ids],full_logits[ids]))
                results.append(dict(fold=fold,task=task,metrics=metrics,inner_models=inner_reports))
                write(out/'metrics.json',results)
                print(f'EXPORTED {fold} {task} OOF and probability control',flush=True)
        manifest.update(status='complete',elapsed_seconds=time.time()-manifest['started_unix'],
                        output_files_sha256={str(p.relative_to(out)):sha(p) for root in ['vectors','in_sample_probability_control'] for p in (out/root).rglob('*.npz')})
        write(out/'run_manifest.json',manifest)
        print('OOF COMPLETE',manifest['completed_models'],'models',flush=True)
    except BaseException:
        manifest.update(status='failed',error=traceback.format_exc())
        write(out/'run_manifest.json',manifest)
        raise


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['shared','original-vectors','checkpoint-manifest','output']:
        p.add_argument('--'+name,required=True)
    for name,default in [('inner-folds',5),('epochs',15),('patience',4),('batch',512),('buckets',2**20),('dim',128),('threads',4),('seed',0)]:
        p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--lr',type=float,default=.01)
    args=p.parse_args()
    if args.inner_folds<2 or any(getattr(args,n)<1 for n in ['epochs','patience','batch','buckets','dim','threads']) or not np.isfinite(args.lr) or args.lr<=0:
        p.error('Invalid training parameters')
    return args


if __name__=='__main__':
    run(parse_args())
