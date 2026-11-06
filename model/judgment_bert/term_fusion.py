"""Term prediction with detached fastText features and ELECTRA-query attention."""
from __future__ import annotations

import math

import torch
from torch import nn

from .model import SingleTaskBert


class FrozenFastTextTermModel(SingleTaskBert):
    """Accept precomputed accusation/article hidden vectors, never label targets.

    fastText extraction happens outside this module, in eval/no_grad mode.
    Detach precedes the trainable projections so term loss can adapt them.
    """

    def __init__(self, num_labels, accusation_dim, article_dim, beta=0.1,
                 attention_dim=128,
                 model_name='hfl/chinese-legal-electra-base-discriminator',
                 revision=None, pooling='cls', head='mlp', finetune='last_n',
                 last_n=2, chunk_pooling='mean', dropout=0.1, encoder=None,
                 learnable_beta=False):
        if any(isinstance(d, bool) or not isinstance(d, int) or d < 1
               for d in (accusation_dim, article_dim, attention_dim)):
            raise ValueError('Feature and attention dimensions must be positive integers')
        if not math.isfinite(beta) or beta < 0:
            raise ValueError('beta must be finite and nonnegative')
        if learnable_beta and not 0 < beta < 1:
            raise ValueError('Learnable beta initial value must be strictly between 0 and 1')
        super().__init__(num_labels, model_name=model_name, revision=revision,
                         pooling=pooling, head=head, finetune=finetune, last_n=last_n,
                         chunk_pooling=chunk_pooling, dropout=dropout, encoder=encoder)
        hidden = self.encoder.config.hidden_size
        self.accusation_dim = accusation_dim
        self.article_dim = article_dim
        self.attention_dim = attention_dim
        # A buffer preserves the fixed hyperparameter in the model state_dict.
        self.register_buffer('beta', torch.tensor(float(beta)))
        if learnable_beta:
            self.beta_logit = nn.Parameter(torch.tensor(math.log(beta / (1 - beta))))
        self.accusation_projection = nn.Sequential(nn.Linear(accusation_dim, hidden), nn.GELU())
        self.article_projection = nn.Sequential(nn.Linear(article_dim, hidden), nn.GELU())
        self.query = nn.Linear(hidden, attention_dim, bias=False)
        self.key = nn.Linear(hidden, attention_dim, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.auxiliary_norm = nn.LayerNorm(hidden, elementwise_affine=False)

    def forward(self, input_ids, attention_mask, accusation_vectors, article_vectors,
                chunk_mask=None, return_attention=False):
        batch = input_ids.shape[0]
        for name, vectors, dim in (
            ('accusation', accusation_vectors, self.accusation_dim),
            ('article', article_vectors, self.article_dim),
        ):
            if vectors.shape != (batch, dim) or not vectors.is_floating_point():
                raise ValueError(f'{name}_vectors must be floating [batch, {dim}]')
            if vectors.device != input_ids.device:
                raise ValueError(f'{name}_vectors and tokens must be on the same device')
            if not torch.isfinite(vectors).all():
                raise ValueError(f'{name}_vectors must be finite')

        term = self.encode(input_ids, attention_mask, chunk_mask)
        return self.fuse_features(term, accusation_vectors.detach(), article_vectors.detach(),
                                  return_attention=return_attention)

    def fuse_features(self, term, accusation_vectors, article_vectors, return_attention=False):
        """Fuse features without detaching; the frozen forward detaches at its boundary."""
        dtype = self.accusation_projection[0].weight.dtype
        sources = torch.stack((
            self.accusation_projection(accusation_vectors.to(dtype=dtype)),
            self.article_projection(article_vectors.to(dtype=dtype)),
        ), dim=1)
        query = self.query(term).unsqueeze(1)
        keys, values = self.key(sources), self.value(sources)
        scores = (query * keys).sum(-1) / math.sqrt(self.attention_dim)
        # Softmax in FP32 protects the two-source weights during mixed precision.
        weights = scores.float().softmax(-1).to(values.dtype)
        auxiliary = (weights.unsqueeze(-1) * values).sum(1)
        fused = term + self.effective_beta() * self.auxiliary_norm(auxiliary)
        logits = self.head(fused)
        if return_attention:
            return logits, weights  # source order: accusation, article
        return logits

    def effective_beta(self):
        """One global scalar, bounded to (0, 1) in the learnable variant."""
        return self.beta_logit.sigmoid() if hasattr(self, 'beta_logit') else self.beta


class TermRegressionHead(nn.Module):
    """Preserve the existing term baseline's seven bins and log-month regression."""
    def __init__(self, hidden, dropout):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(hidden, 512), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout))
        self.bin = nn.Linear(256, 7)
        self.reg = nn.Linear(256, 1)

    def forward(self, features):
        hidden = self.mlp(features)
        return {'bin': self.bin(hidden), 'reg': self.reg(hidden)}


class FrozenFastTextDualTermModel(FrozenFastTextTermModel):
    def __init__(self, accusation_dim, article_dim, **kwargs):
        super().__init__(7, accusation_dim, article_dim, **kwargs)
        self.head = TermRegressionHead(self.encoder.config.hidden_size, kwargs.get('dropout', 0.1))
