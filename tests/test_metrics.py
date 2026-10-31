import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from defect_lab import metrics


def test_auroc_matches_hand_count():
    # 3 positif, 2 negatif. Pasangan (pos > neg): 0.9>{0.1,0.5}=2, 0.5 seri dgn 0.5=0.5,
    # 0.5>0.1=1, 0.4>0.1=1 -> 4.5 / 6.
    s = np.array([0.9, 0.5, 0.4, 0.1, 0.5])
    y = np.array([1, 1, 1, 0, 0])
    assert metrics.auroc(s, y) == pytest.approx(4.5 / 6)
    assert metrics.auroc(s, y) == pytest.approx(roc_auc_score(y, s))


def test_auroc_degenerate():
    assert np.isnan(metrics.auroc(np.array([1.0, 2.0]), np.array([1, 1])))


def test_hist_auroc_close_to_exact():
    rng = np.random.default_rng(0)
    neg = rng.normal(0, 1, 200_000)
    pos = rng.normal(1.5, 1, 5_000)
    s = np.concatenate([neg, pos])
    y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    edges = metrics.quantile_edges(s, 2048)
    idx = metrics.bin_index(s, edges)
    nb = len(edges) - 1
    hp = np.bincount(idx[y == 1], minlength=nb)
    hn = np.bincount(idx[y == 0], minlength=nb)
    assert abs(metrics.auroc_from_hist(hp, hn) - roc_auc_score(y, s)) < 1e-3


def test_aupro_perfect_and_random():
    # Peta sempurna: region selalu di atas semua piksel normal -> PRO=1 di FPR 0.
    maps = np.zeros((2, 32, 32))
    masks = np.zeros((2, 32, 32), dtype=np.uint8)
    masks[0, 2:6, 2:6] = 1
    masks[0, 20:30, 20:30] = 1
    maps[0][masks[0] == 1] = 10.0
    maps += np.random.default_rng(1).uniform(0, 1, maps.shape)
    ph = metrics.pixel_hists(maps, masks, 256)
    assert len(ph.region_hist) == 2
    assert metrics.aupro_from_hist(ph.region_hist, ph.neg.sum(0)) > 0.99
    # Skor acak: PRO(t) kira-kira = FPR(t), sehingga AUPRO sampai 0,3 kira-kira 0,15.
    rnd = np.random.default_rng(2).uniform(0, 1, maps.shape)
    ph = metrics.pixel_hists(rnd, masks, 256)
    assert metrics.aupro_from_hist(ph.region_hist, ph.neg.sum(0)) == pytest.approx(0.15, abs=0.05)


def test_pixel_hist_counts_all_pixels():
    rng = np.random.default_rng(3)
    maps = rng.normal(size=(3, 16, 16))
    masks = (rng.uniform(size=(3, 16, 16)) > 0.9).astype(np.uint8)
    ph = metrics.pixel_hists(maps, masks, 64)
    assert ph.pos.sum() == masks.sum()
    assert ph.neg.sum() == masks.size - masks.sum()
    exact = metrics.auroc(maps.reshape(-1), masks.reshape(-1))
    assert abs(metrics.auroc_from_hist(ph.pos.sum(0), ph.neg.sum(0)) - exact) < 0.01


def test_iou():
    a = np.zeros((4, 4), bool)
    b = np.zeros((4, 4), bool)
    a[:2] = True
    b[1:3] = True
    assert metrics.iou(a, b) == pytest.approx(4 / 12)
    assert metrics.iou(np.zeros((2, 2)), np.zeros((2, 2))) == 1.0
