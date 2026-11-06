"""Train fold-local fastText accusation/article branches with ELECTRA term prediction.

Shared CSV roles are authoritative. All eligible TRAIN cases supervise their
available tasks; only VALID imprisonment CAIL selects the joint checkpoint.
TEST and the fixed 2025 holdout are evaluated once after checkpoint selection.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from transformers import AutoTokenizer

import evaluate as E
from judgment_bert.fasttext_mtl import (FastTextElectraMTL, FastTextGradientMixer, clip_dense_and_sparse_gradients,
                                      masked_mtl_loss)


TASKS = ('imprisonment', 'accusation', 'article')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shared', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--fold', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--patience', type=int, default=2)
    p.add_argument('--last-n', type=int, default=6)
    p.add_argument('--max-len', type=int, default=512)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--encoder-lr', type=float, default=2e-5)
    p.add_argument('--head-lr', type=float, default=1e-3)
    p.add_argument('--fasttext-lr', type=float, default=0.01)
    p.add_argument('--buckets', type=int, default=2**20)
    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--beta', type=float, default=0.2)
    p.add_argument('--accusation-weight', type=float, default=0.1)
    p.add_argument('--article-weight', type=float, default=0.1)
    p.add_argument('--stop-term-gradient', action='store_true')
    p.add_argument('--term-gradient-ratio', type=float,
                   help='Match term-to-fastText gradient norm to this fraction of its weighted own gradient')
    p.add_argument('--fasttext-checkpoints', help='Per-fold pretrained fastText weights and integrity manifest')
    p.add_argument('--term-cohort-only', action='store_true', help='Use the original imprisonment cohort for all branches')
    args = p.parse_args()
    if args.term_gradient_ratio is not None:
        if args.stop_term_gradient or not math.isfinite(args.term_gradient_ratio) or args.term_gradient_ratio < 0:
            p.error('term-gradient-ratio must be finite, nonnegative and requires term backprop')
    for name in ('epochs', 'patience', 'last_n', 'max_len', 'batch', 'grad_accum', 'buckets', 'dim'):
        if getattr(args, name) < 1:
            p.error(f'{name} must be positive')
    for name in ('encoder_lr', 'head_lr', 'fasttext_lr'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            p.error(f'{name} must be positive and finite')
    for name in ('beta', 'accusation_weight', 'article_weight'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            p.error(f'{name} must be nonnegative and finite')
    return args


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare(records, tokenizer, vectorizer, classes, args):
    if args.term_cohort_only:
        records = [r for r in records if r['valid']['imprisonment']]
    records = [r for r in records if any(r['valid'][task] for task in TASKS)]
    if not records:
        raise ValueError('Every split must contain at least one labelled case')
    ids = tokenizer([r['fact'] for r in records], truncation=True, max_length=args.max_len)['input_ids']
    X = vectorizer.transform([r['fact'] for r in records]).tocsr()
    X.sort_indices()
    masks = {task: torch.tensor([r['valid'][task] for r in records]) for task in TASKS}
    months = np.array([r['months'] for r in records])
    # This experiment is finite imprisonment only. Detention stays auxiliary-only.
    if np.any(months[masks['imprisonment'].numpy()] <= 0):
        raise ValueError('Expected positive finite imprisonment months')
    targets = dict(bin=torch.tensor([E.bin_of(m) for m in months]),
                   log_months=torch.tensor(np.log1p(np.clip(months, 0, None)), dtype=torch.float32))
    for task in ('accusation', 'article'):
        mlb = MultiLabelBinarizer(classes=classes[task]).fit([[]])
        targets[task] = torch.tensor(mlb.transform([
            r[task] if r['valid'][task] else [] for r in records]), dtype=torch.float32)
    return dict(records=records, ids=ids, X=X, targets=targets, masks=masks)


def batch_tensors(data, selected, tokenizer, device):
    seqs = [data['ids'][i] for i in selected]
    width = max(map(len, seqs))
    tokens = np.full((len(seqs), width), tokenizer.pad_token_id, dtype=np.int64)
    mask = np.zeros_like(tokens)
    for i, seq in enumerate(seqs):
        tokens[i, :len(seq)] = seq
        mask[i, :len(seq)] = 1
    sub = data['X'][selected]
    arrays = (tokens, mask, sub.indices.astype(np.int64), sub.indptr[:-1].astype(np.int64), sub.data)
    return tuple(torch.from_numpy(a).to(device) for a in arrays)


def evaluate(model, data, tokenizer, device, args, top):
    model.eval()
    outputs = {}
    order = np.argsort([len(s) for s in data['ids']])
    with torch.no_grad(), torch.autocast(device.type, enabled=device.type == 'cuda'):
        for i in range(0, len(order), args.batch * 4):
            out = model(*batch_tensors(data, order[i:i + args.batch * 4], tokenizer, device))
            for key, value in out.items():
                outputs.setdefault(key, []).append(value.float().cpu().numpy())
    back = np.argsort(order)
    outputs = {k: np.concatenate(v)[back] for k, v in outputs.items()}
    term = data['masks']['imprisonment'].numpy()
    if not term.any():
        raise ValueError('Each evaluation split must contain imprisonment cases')
    months_true = np.array([r['months'] for r in data['records']])[term]
    log_prediction = outputs['reg'][term, 0]
    if not np.isfinite(log_prediction).all():
        raise ValueError('Nonfinite regression prediction')
    months_pred = np.clip(np.expm1(log_prediction.astype(np.float64)), 0, None).round()
    if not np.isfinite(months_pred).all():
        raise ValueError('Nonfinite month prediction')
    metrics = E.term_scores(months_true, outputs['bin'][term].argmax(1), months_pred)
    auxiliary = {}
    for task in ('accusation', 'article'):
        valid = data['masks'][task].numpy()
        auxiliary[task] = dict(n=int(valid.sum()), metrics=E.multilabel_scores(
            data['targets'][task].numpy()[valid], E.at_least_one(outputs[task][valid]), top[task])
        ) if valid.any() else dict(n=0, metrics=None)
        if auxiliary[task]['metrics'] is not None:
            auxiliary[task]['metrics'] = {k: v if math.isfinite(v) else None
                                           for k, v in auxiliary[task]['metrics'].items()}
    metrics.update(n_term=int(term.sum()), n_all=len(data['records']), auxiliary=auxiliary,
                   attention_mean=outputs['attention'][term].mean(0).tolist())
    return metrics


def train(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('DEVICE', device, torch.cuda.get_device_name(0) if device.type == 'cuda' else '', flush=True)
    rows, assignments, classes = E.load_shared(args.shared, 'relevant_articles_with_suffix')
    if args.fold not in E.shared_folds(assignments):
        raise ValueError('Unknown fold')
    splits = E.shared_split(rows, assignments, args.fold)
    np.random.RandomState(args.seed).shuffle(splits['train'])
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    vectorizer = HashingVectorizer(analyzer='char', ngram_range=(1, 2), n_features=args.buckets,
                                  alternate_sign=False, norm='l2', dtype=np.float32)
    data = {role: prepare(records, tokenizer, vectorizer, classes, args) for role, records in splits.items()}
    counts = {role: dict(total=len(d['records']), **{t: int(d['masks'][t].sum()) for t in TASKS})
              for role, d in data.items()}
    source_sha256 = {name: hashlib.sha256((Path(args.shared) / name).read_bytes()).hexdigest()
                     for name in ('shared_100k_with_article_suffix.jsonl', 'shared_fold_assignments.csv')}
    top, pos_weights = {}, {}
    for task in ('accusation', 'article'):
        Y = data['train']['targets'][task][data['train']['masks'][task]].numpy()
        if not len(Y):
            raise ValueError(f'No TRAIN labels for {task}')
        top[task] = E.top_classes(Y)
        pos = Y.sum(0)
        pos_weights[task] = torch.tensor(np.clip(np.sqrt((len(Y) - pos) / np.maximum(pos, 1)), 1, 30),
                                         dtype=torch.float32, device=device)
    model = FastTextElectraMTL(len(classes['accusation']), len(classes['article']),
        buckets=args.buckets, dim=args.dim, beta=args.beta, term_backprop=not args.stop_term_gradient,
        model_name=args.model, last_n=args.last_n, dropout=0.1).to(device)
    warm_start = None
    if args.fasttext_checkpoints:
        warm_start = model.load_fasttext_checkpoints(args.fasttext_checkpoints, args.fold, classes)
        manifest = json.loads((Path(args.fasttext_checkpoints) / 'fasttext_checkpoint_manifest.json').read_text())
        if source_sha256 != manifest['data_sha256']:
            raise ValueError('Pretrained fastText belongs to different source data or folds')
        roles_by_line = {int(a['sample_line_1based']): a[args.fold] for a in assignments}
        for task, metadata in warm_start.items():
            if any(roles_by_line.get(line) != 'train' for line in metadata['train_source_lines']):
                raise ValueError('Pretrained fastText reports training outside this fold TRAIN')
        save_json(output / 'fasttext_initial_metadata.json', warm_start)
        print('WARM START', {task: m['selected_epoch'] for task, m in warm_start.items()}, flush=True)
    sparse = model.sparse_parameters()
    mixer = FastTextGradientMixer(sparse, args.term_gradient_ratio) if args.term_gradient_ratio is not None else None
    sparse_ids = {id(p) for p in sparse}
    encoder = [p for p in model.encoder.parameters() if p.requires_grad]
    heads = [p for n, p in model.named_parameters()
             if p.requires_grad and not n.startswith(('encoder.', 'accusation_fasttext.', 'article_fasttext.'))]
    fasttext_heads = [p for n, p in model.named_parameters()
                     if n.startswith(('accusation_fasttext.', 'article_fasttext.')) and id(p) not in sparse_ids]
    dense_opt = torch.optim.AdamW([dict(params=encoder, lr=args.encoder_lr),
                                   dict(params=heads, lr=args.head_lr),
                                   dict(params=fasttext_heads, lr=args.fasttext_lr, weight_decay=0.)], weight_decay=.01)
    sparse_opt = torch.optim.SparseAdam(sparse, lr=args.fasttext_lr)
    optimizers = (dense_opt, sparse_opt)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')
    started = time.time()
    history, best_score, best_epoch, stale = [], -1., 0, 0
    settings = dict(args=vars(args), counts=counts, classes=classes, source_sha256=source_sha256,
                    gradient_mixing='per-backbone accumulated weighted-own + alpha * norm(weighted-own)/norm(term) * term; before clipping and Adam' if mixer else None,
                    fasttext_initialization='per-fold pretrained VALID-best weights' if warm_start else 'random normal std=.05',
                    fasttext_selected_epochs={t: m['selected_epoch'] for t,m in warm_start.items()} if warm_start else None,
                    training_cohort='original imprisonment-valid cohort for all branches' if args.term_cohort_only else 'union of task-valid TRAIN rows; term loss masks out detention',
                    selection='VALID imprisonment CAIL only', term_units='months',
                    vectorizer=dict(analyzer='char', ngram_range=[1, 2], n_features=args.buckets,
                                    alternate_sign=False, norm='l2'))
    save_json(output / 'settings.json', settings)
    print('COUNTS', json.dumps(counts), flush=True)
    tr = data['train']
    for ep in range(1, args.epochs + 1):
        model.train()
        permutation = np.random.permutation(len(tr['records']))
        batches = math.ceil(len(permutation) / args.batch)
        running = dict(total=0., term=0., accusation=0., article=0.)
        gradient_stats = []
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        for step, start in enumerate(range(0, len(permutation), args.batch)):
            selected = permutation[start:start + args.batch]
            with torch.autocast(device.type, enabled=device.type == 'cuda'):
                out = model(*batch_tensors(tr, selected, tokenizer, device))
            masks = {k: v[selected].to(device) for k, v in tr['masks'].items()}
            targets = {k: v[selected].to(device) for k, v in tr['targets'].items()}
            loss, components = masked_mtl_loss(out, targets, masks, pos_weights,
                                               args.accusation_weight, args.article_weight)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            group_start = step - step % args.grad_accum
            divisor = min(args.grad_accum, batches - group_start)
            if mixer:
                own_loss = args.accusation_weight * components['accusation'] + args.article_weight * components['article']
                mixer.accumulate(scaler.scale(own_loss / divisor), scaler.scale(components['term'] / divisor))
            scaler.scale(loss / divisor).backward()
            if (step + 1) % args.grad_accum == 0 or step + 1 == batches:
                if mixer:
                    gradient_stats.append(mixer.apply(scaler.get_scale()))
                for optimizer in optimizers:
                    scaler.unscale_(optimizer)
                if args.stop_term_gradient or mixer:
                    # Separate clipping avoids letting the term gradient norm rescale auxiliary updates.
                    clip_dense_and_sparse_gradients(encoder + heads)
                    clip_dense_and_sparse_gradients(model.accusation_fasttext.parameters())
                    clip_dense_and_sparse_gradients(model.article_fasttext.parameters())
                else:
                    clip_dense_and_sparse_gradients(model.parameters())
                for optimizer in optimizers:
                    scaler.step(optimizer)
                scaler.update()
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
            running['total'] += loss.item() * len(selected)
            for key, value in components.items():
                running[key] += value.item() * len(selected)
            if step % 500 == 0:
                print(f'[{args.fold}] epoch {ep} {start}/{len(permutation)} loss={loss.item():.4f} '
                      f'components={dict((k, round(v.item(), 4)) for k,v in components.items())}', flush=True)
        valid = evaluate(model, data['valid'], tokenizer, device, args, top)
        score = valid['CAIL 分數']
        history.append(dict(epoch=ep, train_batch_mean_losses={k:v/len(permutation) for k,v in running.items()},
                            valid=valid, elapsed_seconds=time.time() - started))
        if mixer:
            history[-1]['gradient_ratios'] = {
                task: dict(steps=len(gradient_stats),
                           finite_steps=sum(s[i]['finite'] for s in gradient_stats),
                           active_steps=sum(s[i]['finite'] and s[i]['own_norm'] > 0 and s[i]['term_norm'] >= 1e-12 for s in gradient_stats),
                           mean_ratio=float(np.mean([s[i]['ratio'] for s in gradient_stats if s[i]['finite']]))
                           if any(s[i]['finite'] for s in gradient_stats) else None)
                for i, task in enumerate(('accusation', 'article'))}
            save_json(output / f'gradient_stats_epoch_{ep}.json', gradient_stats)
        if score > best_score:
            best_score, best_epoch, stale = score, ep, 0
            temporary = output / 'best.pt.tmp'
            torch.save(model.state_dict(), temporary)
            temporary.replace(output / 'best.pt')
        else:
            stale += 1
        save_json(output / 'history.json', history)
        print(f'[{args.fold}] epoch {ep} VALID CAIL={score:.6f} best={best_score:.6f}', flush=True)
        if stale >= args.patience:
            break
    # Free Adam state before restoring the 1+ GB checkpoint on the GPU.
    del optimizers, dense_opt, sparse_opt, scaler
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    model.load_state_dict(torch.load(output / 'best.pt', map_location='cpu', weights_only=True))
    evaluations = [dict(evaluate(model, data[role], tokenizer, device, args, top), 評分=role)
                   for role in ('test', 'holdout')]
    result = dict(fold=args.fold, selected_epoch=best_epoch, best_valid_cail=best_score,
                  args=vars(args), settings=settings, history=history, evaluations=evaluations,
                  elapsed_seconds=time.time() - started)
    tokenizer.save_pretrained(output / 'tokenizer')
    model.encoder.config.save_pretrained(output / 'encoder_config')
    save_json(output / 'metrics.json', result)
    print('FOLD COMPLETE', json.dumps(result['evaluations'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    train(parse_args())
