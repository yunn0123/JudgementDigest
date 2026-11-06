"""Joint fastText accusation/article encoders and ELECTRA term attention.

Only facts enter forward; label targets are used by the losses outside the model.
Auxiliary BCE losses update fastText; term gradients can optionally be blocked.
"""
from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .term_fusion import FrozenFastTextDualTermModel


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class FastTextBranch(nn.Module):
    """Match models/fasttext_gpu.py: weighted sum, dropout, linear logits."""
    def __init__(self, buckets, dim, num_labels):
        super().__init__()
        if min(buckets, dim, num_labels) < 1:
            raise ValueError('FastText dimensions and label count must be positive')
        self.emb = nn.EmbeddingBag(buckets, dim, mode='sum', sparse=True)
        nn.init.normal_(self.emb.weight, std=0.05)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(dim, num_labels)

    def forward(self, indices, offsets, weights):
        features = self.emb(indices, offsets, per_sample_weights=weights)
        return features, self.head(self.drop(features))


class FastTextElectraMTL(FrozenFastTextDualTermModel):
    """Trainable heterogeneous backbones; reuse the frozen model's fusion and head.

    Despite the inherited class name, fastText embeddings are trainable here.
    Auxiliary heads use fastText features directly; ELECTRA supplies the query
    for term fusion and is updated only by the term loss.
    """
    def __init__(self, accusation_labels, article_labels, buckets=2**20, dim=128,
                 beta=0.2, term_backprop=True, **kwargs):
        super().__init__(dim, dim, beta=beta, **kwargs)
        self.term_backprop = term_backprop
        self.accusation_fasttext = FastTextBranch(buckets, dim, accusation_labels)
        self.article_fasttext = FastTextBranch(buckets, dim, article_labels)

    def forward(self, input_ids, attention_mask, indices, offsets, weights):
        accusation, accusation_logits = self.accusation_fasttext(indices, offsets, weights)
        article, article_logits = self.article_fasttext(indices, offsets, weights)
        term = self.encode(input_ids, attention_mask)
        term_accusation = accusation if self.term_backprop else accusation.detach()
        term_article = article if self.term_backprop else article.detach()
        prediction, attention = self.fuse_features(term, term_accusation, term_article, return_attention=True)
        return dict(prediction, accusation=accusation_logits, article=article_logits,
                    attention=attention)

    def sparse_parameters(self):
        return [self.accusation_fasttext.emb.weight, self.article_fasttext.emb.weight]

    def load_fasttext_checkpoints(self, folder, fold, classes):
        """Load the original per-fold best models; enforce class order and integrity."""
        folder = Path(folder)
        manifest = json.loads((folder / 'fasttext_checkpoint_manifest.json').read_text())
        loaded = {}
        for task in ('accusation', 'article'):
            relative = f'{fold}/{task}.pt'
            checkpoint = folder / relative
            if file_sha256(checkpoint) != manifest['sha256'][relative]:
                raise ValueError('fastText checkpoint checksum mismatch')
            metadata_path = folder / fold / f'{task}.json'
            if hashlib.sha256(metadata_path.read_bytes()).hexdigest() != manifest['sha256'][f'{fold}/{task}.json']:
                raise ValueError('fastText metadata checksum mismatch')
            metadata = json.loads(metadata_path.read_text())
            if metadata['fold'] != fold or metadata['task'] != task or metadata['classes'] != classes[task]:
                raise ValueError('fastText fold/task/class order mismatch')
            if task == 'article' and metadata['article_field'] != 'relevant_articles_with_suffix':
                raise ValueError('Expected suffix-preserving fastText checkpoint')
            branch = getattr(self, task + '_fasttext')
            vectorizer = metadata['vectorizer']
            if (metadata['feature_dim'] != branch.emb.embedding_dim
                    or vectorizer['n_features'] != branch.emb.num_embeddings
                    or vectorizer['analyzer'] != 'char' or list(vectorizer['ngram_range']) != [1, 2]
                    or vectorizer['norm'] != 'l2' or vectorizer['alternate_sign']):
                raise ValueError('fastText feature configuration mismatch')
            state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
            branch.load_state_dict({'emb.weight': state['emb.weight'],
                                    'head.weight': state['heads.out.weight'],
                                    'head.bias': state['heads.out.bias']}, strict=True)
            loaded[task] = metadata
        return loaded


def masked_mtl_loss(output, targets, masks, pos_weights,
                    accusation_weight=0.1, article_weight=0.1):
    """Unlabelled tasks are masked out, never interpreted as negative labels.

    Term loss = CE(seven bins) + MSE(log1p months); auxiliary losses are means
    over valid samples and classes, following the original fastText training.
    """
    if any(not math.isfinite(w) or w < 0 for w in (accusation_weight, article_weight)):
        raise ValueError('Auxiliary loss weights must be finite and nonnegative')
    components = {}
    valid = masks['imprisonment'].bool()
    components['term'] = (F.cross_entropy(output['bin'][valid].float(), targets['bin'][valid])
                          + F.mse_loss(output['reg'][valid, 0].float(), targets['log_months'][valid])) if valid.any() else output['bin'].float().sum() * 0
    for task in ('accusation', 'article'):
        valid = masks[task].bool()
        components[task] = F.binary_cross_entropy_with_logits(
            output[task][valid].float(), targets[task][valid].float(),
            pos_weight=pos_weights[task],
        ) if valid.any() else output[task].float().sum() * 0
    loss = components['term'] + accusation_weight * components['accusation'] + article_weight * components['article']
    return loss, components


class FastTextGradientMixer:
    """Match term gradient norms to each branch's weighted own gradient per optimizer step.

    Accumulate before normalization so the ratio applies to the effective batch,
    not independently to microbatches whose gradients may cancel.
    """
    def __init__(self, parameters, alpha):
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError('alpha must be finite and nonnegative')
        self.parameters = list(parameters)
        self.alpha = alpha
        self.own = self.term = None

    def accumulate(self, scaled_own_loss, scaled_term_loss):
        for name, loss in (('own', scaled_own_loss), ('term', scaled_term_loss)):
            grads = torch.autograd.grad(loss, self.parameters, retain_graph=True)
            previous = getattr(self, name)
            values = []
            for i, grad in enumerate(grads):
                grad = grad.detach()
                if not grad.is_sparse:
                    raise ValueError('Expected sparse fastText embedding gradients')
                values.append((grad if previous is None else previous[i] + grad).coalesce())
            setattr(self, name, values)

    @torch.no_grad()
    def apply(self, loss_scale=1.0):
        if self.own is None or self.term is None:
            raise RuntimeError('No accumulated task gradients')
        stats = []
        for parameter, own, term in zip(self.parameters, self.own, self.term):
            own_norm = own.values().float().norm() / loss_scale
            term_norm = term.values().float().norm() / loss_scale
            if not (torch.isfinite(own_norm) and torch.isfinite(term_norm)):
                # Preserve nonfinite gradients for AMP's overflow detection and skipped step.
                parameter.grad = (own + term).coalesce()
                stats.append(dict(finite=False))
                continue
            factor = self.alpha * own_norm / term_norm.clamp_min(1e-12)
            contribution = term * factor
            parameter.grad = (own + contribution).coalesce()
            contribution_norm = contribution.coalesce().values().float().norm() / loss_scale
            stats.append(dict(finite=True, own_norm=own_norm.item(), term_norm=term_norm.item(),
                              term_factor=factor.item(), contribution_norm=contribution_norm.item(),
                              ratio=(contribution_norm / own_norm).item() if own_norm > 0 else 0.))
        self.own = self.term = None
        return stats


def clip_dense_and_sparse_gradients(parameters, max_norm=1.0):
    """Global clipping after AMP unscale; torch clip_grad_norm_ rejects sparse grads."""
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError('max_norm must be positive and finite')
    gradients = []
    for parameter in parameters:
        if parameter.grad is None:
            continue
        if parameter.grad.is_sparse:
            parameter.grad = parameter.grad.coalesce()
            gradients.append(parameter.grad.values())
        else:
            gradients.append(parameter.grad)
    if not gradients:
        return torch.tensor(0.)
    norm = torch.stack([g.detach().float().norm() for g in gradients]).norm()
    scale = (max_norm / (norm + 1e-6)).clamp(max=1.0)
    for gradient in gradients:
        gradient.mul_(scale.to(gradient.dtype))
    return norm
