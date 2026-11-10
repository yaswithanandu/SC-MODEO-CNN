"""Decoder checks: every random genome must build, run forward/backward and report MFLOPs."""
import os, random, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modeo.encoding import Genome, random_genome
from modeo.model import build_model
from modeo.losses import make_loss


def test_random_genomes_build_and_run():
    rng = random.Random(1)
    for n_feat in (10, 26, 128):
        for _ in range(40):
            g = random_genome(3, 10, rng, use_loss_gene=True)
            m = build_model(g, n_feat, 8)
            x = torch.randn(5, n_feat)
            logits, z = m.forward_with_features(x)
            assert logits.shape == (5, 8) and z.shape == (5, m.embed_dim)
            loss = make_loss(g.setting.loss_name, np.array([50, 5, 5, 5, 5, 5, 5, 1]), "cpu")(logits, torch.randint(0, 8, (5,)))
            loss.backward()
            mf = m.mflops()
            assert np.isfinite(mf) and mf > 0


def test_mflops_grows_with_width():
    small = Genome.from_list([3, "111", [0, 0, 6, 2, 0], [0, 0, 6, 2, 0], [0, 0, 6, 2, 0], [0, 1, 3, 0]])
    big = Genome.from_list([3, "111", [7, 3, 6, 2, 0], [7, 3, 6, 2, 0], [7, 3, 6, 2, 0], [0, 1, 3, 0]])
    assert build_model(big, 26, 8).mflops() > 10 * build_model(small, 26, 8).mflops()


def test_mflops_matches_hand_count():
    # 1 block: 16 channels, k=3, no pooling, linear act, L=10 -> conv 2*1*16*3*10 + bias 160 + act 160
    g = Genome.from_list([3, "000", [0, 0, 7, 2, 0], [0, 0, 7, 2, 0], [0, 0, 7, 2, 0], [0, 1, 3, 0]])
    m = build_model(g, 10, 2)
    per_first = 2 * 1 * 16 * 3 * 10 + 160 + 160
    per_next = 2 * 16 * 16 * 3 * 10 + 160 + 160
    expected = per_first + 2 * per_next + 160 + 2 * 16 * 2  # + GAP + head
    assert abs(m.mflops() * 1e6 - expected) < 1e-6


def test_mflops_follows_model_device():
    """Every conv must see its input on the same device as its weights ('meta' stands in for CUDA)."""
    rng = random.Random(3)

    def check(mod, args):
        assert args[0].device == mod.weight.device, f"input on {args[0].device}, weights on {mod.weight.device}"

    for _ in range(10):
        g = random_genome(3, 10, rng, use_loss_gene=True)
        cpu = build_model(g, 18, 7).mflops()
        m = build_model(g, 18, 7).to("meta")
        for blk in m.blocks:
            blk.conv.register_forward_pre_hook(check)
        assert m.mflops() == cpu
