"""Metrik deteksi dan lokalisasi cacat.

- image AUROC: dari skor per citra (eksak).
- pixel AUROC dan AUPRO: dihitung dari histogram skor per citra/region dengan tepi bin
  kuantil per kelas. Histogram membuat bootstrap murah; selisihnya terhadap AUROC eksak
  diuji di tests/test_metrics.py dan dilaporkan di evaluation/results.json.
- IoU per citra cacat pada ambang operasi, lalu dirata-rata (mIoU).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.stats import rankdata


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC eksak via peringkat (Mann-Whitney), nilai seri dihitung setengah."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(scores, method="average")
    return float((r[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auroc_from_hist(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUROC dari histogram (bin naik). Pasangan dalam bin yang sama dihitung setengah."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    tp, tn = pos.sum(), neg.sum()
    if tp == 0 or tn == 0:
        return float("nan")
    neg_below = np.concatenate([[0.0], np.cumsum(neg)[:-1]])
    return float(((pos * neg_below).sum() + 0.5 * (pos * neg).sum()) / (tp * tn))


def quantile_edges(values: np.ndarray, bins: int) -> np.ndarray:
    """Tepi bin dari kuantil; nilai unik agar bin tidak kosong lebar nol."""
    q = np.quantile(values, np.linspace(0, 1, bins + 1))
    q = np.unique(q)
    q[0], q[-1] = -np.inf, np.inf
    return q


def bin_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)


@dataclass
class PixelHists:
    """Histogram skor piksel per citra test, untuk pixel AUROC dan AUPRO."""

    edges: np.ndarray
    pos: np.ndarray  # [N, B] piksel cacat
    neg: np.ndarray  # [N, B] piksel normal
    region_hist: np.ndarray  # [R, B] histogram per komponen terhubung cacat
    region_image: np.ndarray  # [R] indeks citra pemilik region


def pixel_hists(maps: np.ndarray, masks: np.ndarray, bins: int, seed: int = 0) -> PixelHists:
    rng = np.random.default_rng(seed)
    flat = maps.reshape(-1)
    sample = flat if flat.size <= 2_000_000 else rng.choice(flat, 2_000_000, replace=False)
    edges = quantile_edges(sample, bins)
    nb = len(edges) - 1
    n = len(maps)
    pos = np.zeros((n, nb), dtype=np.int64)
    neg = np.zeros((n, nb), dtype=np.int64)
    regions, owners = [], []
    for i in range(n):
        idx = bin_index(maps[i].reshape(-1), edges)
        m = masks[i].reshape(-1).astype(bool)
        pos[i] = np.bincount(idx[m], minlength=nb)
        neg[i] = np.bincount(idx[~m], minlength=nb)
        if m.any():
            count, lab = cv2.connectedComponents(masks[i].astype(np.uint8), connectivity=8)
            lab = lab.reshape(-1)
            for r in range(1, count):
                regions.append(np.bincount(idx[lab == r], minlength=nb))
                owners.append(i)
    region_hist = np.array(regions, dtype=np.int64).reshape(-1, nb)
    return PixelHists(edges, pos, neg, region_hist, np.array(owners, dtype=np.int64))


def aupro_from_hist(region_hist: np.ndarray, neg: np.ndarray, fpr_limit: float = 0.3) -> float:
    """AUPRO (Bergmann dkk.) sampai FPR `fpr_limit`, dinormalisasi ke 0..1.

    Ambang disapu dari bin tertinggi ke terendah. PRO = rata-rata (per region) bagian
    piksel region di atas ambang. FPR = bagian piksel normal di atas ambang.
    """
    if len(region_hist) == 0 or neg.sum() == 0:
        return float("nan")
    reg = region_hist.astype(np.float64)
    above_reg = np.cumsum(reg[:, ::-1], axis=1) / reg.sum(1, keepdims=True)
    pro = np.concatenate([[0.0], above_reg.mean(0)])
    fpr = np.concatenate([[0.0], np.cumsum(neg[::-1].astype(np.float64)) / neg.sum()])
    keep = fpr <= fpr_limit
    x, y = fpr[keep], pro[keep]
    if x[-1] < fpr_limit:
        j = int(keep.sum())
        if j < len(fpr):
            t = (fpr_limit - fpr[j - 1]) / (fpr[j] - fpr[j - 1])
            x = np.append(x, fpr_limit)
            y = np.append(y, pro[j - 1] + t * (pro[j] - pro[j - 1]))
    return float(np.trapezoid(y, x) / fpr_limit)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def bootstrap_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.nanquantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
