"""Decode a Genome into a variable-length, blocks-based 1-D CNN and count its MFLOPs.

Assumptions (the paper does not specify these; documented in README):
  * tabular features are treated as a 1-channel 1-D signal of length F (Conv1d);
  * a block with several predecessors concatenates them on the channel axis
    (DenseNet-style), after adaptive-average-pooling them to the shortest length;
  * convolutions use 'same' padding; pooling uses stride 2 (length is halved)
    and is skipped once the sequence length reaches 1;
  * the classifier head is global-average-pooling + one Linear layer; the pooled
    vector is the penultimate embedding used for zero-day scoring.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import ConvGene, Genome


class _Softmax(nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=1)  # over channels


def make_activation(name: str) -> nn.Module:
    return {
        "softplus": nn.Softplus(),
        "softsign": nn.Softsign(),
        "elu": nn.ELU(),
        "softmax": _Softmax(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "relu": nn.ReLU(),
        "linear": nn.Identity(),
    }[name]


class ConvBlock(nn.Module):
    """One CNN block: Conv1d -> activation -> (optional) pooling."""

    def __init__(self, in_ch: int, gene: ConvGene):
        super().__init__()
        self.gene = gene
        k = gene.kernel_size
        self.conv = nn.Conv1d(in_ch, gene.channels, k, padding=k // 2)
        self.act = make_activation(gene.activation)

    def forward(self, x):
        x = self.act(self.conv(x))
        if self.gene.pooling != "none" and x.shape[-1] > 1:
            ps = self.gene.pool_size  # odd (3,5,7,9) so L + 2*(ps//2) >= ps for any L >= 1
            if self.gene.pooling == "max":
                x = F.max_pool1d(x, ps, stride=2, padding=ps // 2)
            else:
                x = F.avg_pool1d(x, ps, stride=2, padding=ps // 2, count_include_pad=False)
        return x


class BlockCNN(nn.Module):
    def __init__(self, genome: Genome, n_features: int, n_classes: int):
        super().__init__()
        self.genome = genome
        self.n_features = n_features
        self.preds = genome.predecessors()
        out_ch: List[int] = []
        blocks = []
        for k, gene in enumerate(genome.convs):
            in_ch = 1 if k == 0 else sum(out_ch[p] for p in self.preds[k])
            blocks.append(ConvBlock(in_ch, gene))
            out_ch.append(gene.channels)
        self.blocks = nn.ModuleList(blocks)
        self.embed_dim = out_ch[-1]
        self.head = nn.Linear(self.embed_dim, n_classes)

    @staticmethod
    def _merge(tensors: List[torch.Tensor]) -> torch.Tensor:
        if len(tensors) == 1:
            return tensors[0]
        L = min(t.shape[-1] for t in tensors)
        tensors = [t if t.shape[-1] == L else F.adaptive_avg_pool1d(t, L) for t in tensors]
        return torch.cat(tensors, dim=1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate embedding (global-average-pooled output of the last block)."""
        h = x.unsqueeze(1) if x.dim() == 2 else x
        outs: List[torch.Tensor] = []
        for k, blk in enumerate(self.blocks):
            inp = h if k == 0 else self._merge([outs[p] for p in self.preds[k]])
            outs.append(blk(inp))
        return outs[-1].mean(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.features(x)
        return self.head(z), z

    # ---- complexity ---------------------------------------------------------
    @torch.no_grad()
    def mflops(self) -> float:
        """Analytical FLOPs (multiply+add counted as 2) of all blocks + head, in millions."""
        total = 0.0
        h = torch.zeros(1, 1, self.n_features, device=next(self.parameters()).device)
        outs = []
        for k, blk in enumerate(self.blocks):
            inp = h if k == 0 else self._merge([outs[p] for p in self.preds[k]])
            if k > 0 and len(self.preds[k]) > 1:  # adaptive pooling of merged inputs
                total += sum(outs[p].numel() for p in self.preds[k])
            c_in, L = inp.shape[1], inp.shape[-1]
            g = blk.gene
            total += 2.0 * c_in * g.channels * g.kernel_size * L + g.channels * L  # conv + bias
            total += g.channels * L  # activation
            out = blk(inp)
            if g.pooling != "none":
                total += g.channels * out.shape[-1] * g.pool_size
            outs.append(out)
        total += outs[-1].numel()  # global average pooling
        total += 2.0 * self.embed_dim * self.head.out_features
        return total / 1e6


def build_model(genome: Genome, n_features: int, n_classes: int) -> BlockCNN:
    return BlockCNN(genome, n_features, n_classes)
