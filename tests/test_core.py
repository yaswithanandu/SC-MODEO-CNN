"""Fast checks of the torch-free components."""
import os, random, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modeo.encoding import Genome, mutate_population, random_genome
from modeo.pareto import crowding_distance, hypervolume, non_dominated_sort, update_archive
from modeo.ood import auroc, fpr_at_95tpr, Mahalanobis, energy_score, msp_score


def test_encoding_matches_paper_example():
    # Fig. 2 example: [4,110101,[5,0,2,0,2],[0,0,1,2,0],[5,3,5,0,0],[6,3,5,0,1],[2,3,5,0]]
    g = Genome.from_list([4, "110101", [5, 0, 2, 0, 2], [0, 0, 1, 2, 0], [5, 3, 5, 0, 0], [6, 3, 5, 0, 1], [2, 3, 5, 0]])
    assert len(g.topo) == 4 * 3 // 2
    c = g.convs[0]
    assert (c.channels, c.kernel_size, c.activation, c.pooling, c.pool_size) == (96, 3, "elu", "max", 7)
    assert g.convs[1].pooling == "none" and g.convs[2].activation == "tanh"
    s = g.setting
    assert (s.batch_size(256), s.optimizer, s.learning_rate, s.regularizer) == (768, "rmsprop", 1.5e-3, "none")
    assert g.predecessors() == [[], [0], [0, 1], [2]]  # matches the drawing in Fig. 2


def test_mutation_population_size_and_lengths():
    rng = random.Random(0)
    for ulg in (False, True):
        for _ in range(50):
            sn = random_genome(3, 10, rng, use_loss_gene=ulg)
            pop = mutate_population(sn, 3, 10, rng)
            assert len(pop) == sn.n + 3
            for g in pop:
                assert 3 <= g.n <= 10 and len(g.topo) == g.n * (g.n - 1) // 2 and len(g.convs) == g.n
                assert len(g.setting.as_list(ulg)) == (5 if ulg else 4)
            # SC2..SC(n+3) differ from SN in exactly one part
            assert pop[1].convs == sn.convs and pop[-1].convs == sn.convs and pop[-1].topo == sn.topo
            for i in range(sn.n):
                assert pop[i + 2].topo == sn.topo and pop[i + 2].setting == sn.setting


def test_non_dominated_sort_and_crowding():
    F = np.array([[0.1, 5], [0.2, 3], [0.3, 1], [0.25, 4], [0.4, 6]])
    fronts = non_dominated_sort(F)
    assert sorted(fronts[0]) == [0, 1, 2] and sorted(fronts[1]) == [3] and fronts[2] == [4]
    cd = crowding_distance(F[[0, 1, 2]])
    assert np.isinf(cd[0]) and np.isinf(cd[2]) and np.isclose(cd[1], (0.3 - 0.1) + (5 - 1))


class _G:
    def __init__(self, f, k): self.fitness, self._k = f, k
    def key(self): return self._k


def test_archive_respects_amax_and_dominance():
    rng = random.Random(0)
    A = []
    for i in range(60):
        e = rng.random(); A = update_archive(A, _G([e, 1 - e + 0.01 * rng.random()], str(i)), 5, rng)
    assert len(A) <= 5
    F = np.array([g.fitness for g in A])
    assert len(non_dominated_sort(F)[0]) == len(A)
    A2 = update_archive(A, _G([2.0, 2.0], "dominated"), 5, rng)  # dominated point is rejected
    assert all(g.key() != "dominated" for g in A2)


def test_hypervolume():
    assert np.isclose(hypervolume(np.array([[0.5, 0.5]]), [1, 1]), 0.25)
    assert np.isclose(hypervolume(np.array([[0.2, 0.8], [0.8, 0.2]]), [1, 1]), 0.16 + 0.16 - 0.04)
    assert np.isclose(hypervolume(np.array([[0.5, 0.5, 0.5]]), [1, 1, 1]), 0.125)


def test_ood_scores_and_metrics():
    rng = np.random.default_rng(0)
    Zk = np.r_[rng.normal(0, 1, (200, 4)), rng.normal(5, 1, (200, 4))]
    yk = np.r_[np.zeros(200, int), np.ones(200, int)]
    Zu = rng.normal(-6, 1, (100, 4))
    m = Mahalanobis().fit(Zk, yk, 2)
    assert auroc(m.score(Zk), m.score(Zu)) > 0.99
    assert fpr_at_95tpr(m.score(Zk), m.score(Zu)) < 0.05
    assert (m.nearest_class(Zk) == yk).mean() > 0.99
    logits_conf, logits_flat = np.array([[10.0, 0.0]]), np.array([[0.1, 0.0]])
    assert msp_score(logits_flat)[0] > msp_score(logits_conf)[0]
    assert energy_score(logits_flat)[0] > energy_score(logits_conf)[0]


def test_score_fusion_uses_both_scores():
    from modeo.ood import ScoreFusion
    rng = np.random.default_rng(0)
    # known: both scores low; unknown A: only score 1 high; unknown B: only score 2 high
    k1, k2 = rng.gamma(2, 1, 500), rng.gamma(2, 1, 500)
    f = ScoreFusion().fit([k1[:250], k2[:250]])
    ka = f.score([k1[250:], k2[250:]])
    ua = f.score([rng.gamma(2, 1, 200) * 20, rng.gamma(2, 1, 200)])
    ub = f.score([rng.gamma(2, 1, 200), rng.gamma(2, 1, 200) * 20])
    assert auroc(ka, ua) > 0.85 and auroc(ka, ub) > 0.85   # fused score catches both kinds


def test_per_class_threshold_protects_rare_class():
    from modeo.ood import open_set_report
    rng = np.random.default_rng(0)
    # class 0 common with low scores, class 1 rare with naturally high scores; unknowns score higher still
    s_cal = np.r_[rng.normal(0, 1, 950), rng.normal(4, 0.5, 50)]; p_cal = np.r_[np.zeros(950, int), np.ones(50, int)]
    s_k = np.r_[rng.normal(0, 1, 950), rng.normal(4, 0.5, 50)]; y_k = p_k = np.r_[np.zeros(950, int), np.ones(50, int)]
    s_u = rng.normal(7, 0.5, 200); p_u = np.ones(200, int)
    g = open_set_report(s_cal, s_k, s_u, y_k, p_k, p_u, 2)
    c = open_set_report(s_cal, s_k, s_u, y_k, p_k, p_u, 2, pred_known_val=p_cal)
    assert c["open_set_macro_f1"] > g["open_set_macro_f1"] + 0.1
    assert c["unknown_detection_rate"] > 0.9
