"""Hybrid variable-length encoding of MODEO-CNN (Lu et al., IEEE TDSC 2025, Sec. IV-B).

A candidate is  [n, topo, conv_1, ..., conv_n, setting]
  n       : number of CNN blocks, in [C_L, C_H]
  topo    : n*(n-1)/2 binary genes describing block-to-block connections
  conv_i  : 5 integers (knum, ksize, kact, pt, ps)                  -- Eq. (1)
  setting : 4 integers (bs, opt, lr, reg)                            -- Eq. (4)
            SC-MODEO-CNN appends a 5th gene `loss` (extension #5)

Gene -> real value mappings follow Eqs. (1)-(6) of the paper.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import List

# ----- gene ranges (inclusive upper bounds), Eq. (1) and Eq. (4) -----
CONV_RANGES = {"knum": 7, "ksize": 3, "kact": 7, "pt": 2, "ps": 3}
SETTING_RANGES = {"bs": 3, "opt": 3, "lr": 7, "reg": 3}
LOSS_RANGE = 4  # SC extension: 0 CE, 1 weighted CE, 2 class-balanced CE, 3 focal g=1, 4 focal g=2

# ----- decoding tables, Eqs. (2), (3), (5), (6) -----
ACTIVATIONS = ["softplus", "softsign", "elu", "softmax", "sigmoid", "tanh", "relu", "linear"]
POOLINGS = ["max", "mean", "none"]
OPTIMIZERS = ["sgd", "adam", "adadelta", "rmsprop"]
REGULARIZERS = ["none", "l1", "l2", "l1l2"]
LOSSES = ["ce", "weighted_ce", "cb_ce", "focal1", "focal2"]


@dataclass
class ConvGene:
    knum: int
    ksize: int
    kact: int
    pt: int
    ps: int

    # decoded values
    @property
    def channels(self) -> int:
        return 16 * (self.knum + 1)

    @property
    def kernel_size(self) -> int:
        return 2 * self.ksize + 3

    @property
    def activation(self) -> str:
        return ACTIVATIONS[self.kact]

    @property
    def pooling(self) -> str:
        return POOLINGS[self.pt]

    @property
    def pool_size(self) -> int:
        return 2 * self.ps + 3

    def as_list(self) -> List[int]:
        return [self.knum, self.ksize, self.kact, self.pt, self.ps]


@dataclass
class SettingGene:
    bs: int
    opt: int
    lr: int
    reg: int
    loss: int = 0  # only searched when use_loss_gene=True

    def batch_size(self, base: int) -> int:
        return base * (self.bs + 1)

    @property
    def learning_rate(self) -> float:
        return (self.lr + 1) * 2.5e-4

    @property
    def optimizer(self) -> str:
        return OPTIMIZERS[self.opt]

    @property
    def regularizer(self) -> str:
        return REGULARIZERS[self.reg]

    @property
    def loss_name(self) -> str:
        return LOSSES[self.loss]

    def as_list(self, use_loss_gene: bool) -> List[int]:
        base = [self.bs, self.opt, self.lr, self.reg]
        return base + [self.loss] if use_loss_gene else base


@dataclass
class Genome:
    n: int
    topo: List[int]
    convs: List[ConvGene]
    setting: SettingGene
    use_loss_gene: bool = False
    # filled by the search
    fitness: List[float] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    # ---- topology -------------------------------------------------------
    def predecessors(self) -> List[List[int]]:
        """Return, for each block k (0-indexed), the list of earlier blocks feeding it.

        The paper maps topo to a triangular matrix A where a_{i,j}=1 links module i to
        module j+1. Bits are ordered by *source* block, as in Fig. 2 of the paper:
        topo "110101" for n=4 means 1->2, 1->3, 1->4, 2->3, 2->4, 3->4 = 1,1,0,1,0,1.
        Block 1 always reads the network input. If a block has no selected
        predecessor it falls back to the immediately preceding block, so that every
        decoded network is a connected DAG (assumption; the paper does not say how
        isolated blocks are handled). The last block feeds the classifier.
        """
        chosen: List[List[int]] = [[] for _ in range(self.n)]
        idx = 0
        for i in range(self.n - 1):          # source block i
            for j in range(i + 1, self.n):   # target block j
                if self.topo[idx] == 1:
                    chosen[j].append(i)
                idx += 1
        return [[]] + [chosen[k] if chosen[k] else [k - 1] for k in range(1, self.n)]

    # ---- serialisation --------------------------------------------------
    def as_list(self) -> list:
        return [self.n, "".join(map(str, self.topo))] + [c.as_list() for c in self.convs] + [
            self.setting.as_list(self.use_loss_gene)
        ]

    def key(self) -> str:
        return hashlib.md5(json.dumps(self.as_list()).encode()).hexdigest()

    def to_dict(self) -> dict:
        return {"genome": self.as_list(), "fitness": self.fitness, "info": self.info}

    @staticmethod
    def from_list(lst: list, use_loss_gene: bool = False) -> "Genome":
        n = int(lst[0])
        topo = [int(ch) for ch in str(lst[1])]
        convs = [ConvGene(*map(int, c)) for c in lst[2 : 2 + n]]
        s = list(map(int, lst[2 + n]))
        setting = SettingGene(*s[:4], loss=s[4] if len(s) > 4 else 0)
        return Genome(n, topo, convs, setting, use_loss_gene=use_loss_gene or len(s) > 4)

    def clone(self) -> "Genome":
        g = copy.deepcopy(self)
        g.fitness, g.info = [], {}
        return g


# ---- random generators (Algorithm 3) -----------------------------------
def random_conv(rng: random.Random) -> ConvGene:
    return ConvGene(**{k: rng.randint(0, hi) for k, hi in CONV_RANGES.items()})


def random_topo(n: int, rng: random.Random) -> List[int]:
    return [rng.randint(0, 1) for _ in range(n * (n - 1) // 2)]


def random_setting(rng: random.Random, use_loss_gene: bool) -> SettingGene:
    s = SettingGene(**{k: rng.randint(0, hi) for k, hi in SETTING_RANGES.items()})
    if use_loss_gene:
        s.loss = rng.randint(0, LOSS_RANGE)
    return s


def random_genome(c_low: int, c_high: int, rng: random.Random, use_loss_gene: bool = False,
                  n: int | None = None) -> Genome:
    """Algorithm 3: candidate initialisation."""
    n = rng.randint(c_low, c_high) if n is None else n
    return Genome(
        n=n,
        topo=random_topo(n, rng),
        convs=[random_conv(rng) for _ in range(n)],
        setting=random_setting(rng, use_loss_gene),
        use_loss_gene=use_loss_gene,
    )


# ---- Algorithm 4: variable-length discrete mutation ---------------------
def mutate_population(sn: Genome, c_low: int, c_high: int, rng: random.Random) -> List[Genome]:
    """Produce the n+3 candidates SC_1..SC_{n+3} from S_N.

    SC_1        : full re-initialisation with a random n          (mutating gene n)
    SC_2        : topo replaced                                  (mutating topo)
    SC_3..n+2   : one conv block replaced each                   (mutating conv_i)
    SC_{n+3}    : setting replaced (includes the loss gene in SC mode)
    """
    pop = [random_genome(c_low, c_high, rng, sn.use_loss_gene)]
    sc2 = sn.clone()
    sc2.topo = random_topo(sn.n, rng)
    pop.append(sc2)
    for i in range(sn.n):
        sci = sn.clone()
        sci.convs[i] = random_conv(rng)
        pop.append(sci)
    scs = sn.clone()
    scs.setting = random_setting(rng, sn.use_loss_gene)
    pop.append(scs)
    for j, g in enumerate(pop):
        g.info["mutation"] = "SC1_reinit" if j == 0 else "SC2_topo" if j == 1 else (
            "setting" if j == len(pop) - 1 else f"conv{j - 1}")
    return pop
