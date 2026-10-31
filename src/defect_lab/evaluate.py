"""Fit dan evaluasi satu metode pada satu kelas, lalu simpan keluaran mentah.

Keluaran per (metode, kelas) di evaluation/raw/<metode>/<kelas>.npz:
  skor citra test dan validasi, label, ambang operasi, IoU per citra cacat,
  histogram piksel per citra (untuk bootstrap), histogram region (AUPRO).
Bank memori disimpan di models/<metode>/<kelas>.pt untuk ekspor web dan ONNX.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from . import classical, config, metrics
from .data import ClassData, load_class
from .patchcore import NearestHead, PatchEmbedder, embed_images, greedy_coreset, score_embeddings


@dataclass
class Fitted:
    head: NearestHead
    norm_mean: torch.Tensor | None  # standardisasi fitur klasik
    norm_std: torch.Tensor | None
    fit_patches: int
    seconds_fit: float


def load_embedder(method: config.MethodConfig) -> PatchEmbedder | None:
    if method.kind != "patchcore":
        return None
    return PatchEmbedder(method.backbone, method.embed_dim).to(config.DEVICE)


def embed(method: config.MethodConfig, model, images: np.ndarray, fitted: Fitted | None = None):
    if method.kind == "classical":
        x = torch.from_numpy(classical.describe_batch(images))
        if fitted is not None:
            x = (x - fitted.norm_mean) / fitted.norm_std
        return x
    return embed_images(model, images)


def fit(method: config.MethodConfig, model, cd: ClassData) -> Fitted:
    t0 = time.perf_counter()
    imgs, _, _ = cd.select("fit")
    emb = embed(method, model, imgs)
    mean = std = None
    if method.kind == "classical":
        flat = emb.reshape(-1, emb.shape[-1])
        mean, std = flat.mean(0), flat.std(0).clamp_min(1e-6)
        emb = (emb - mean) / std
    bank = emb.reshape(-1, emb.shape[-1])
    m = min(int(math.ceil(method.coreset_ratio * len(bank))), method.coreset_max)
    seed = config.SEED + config.CLASSES.index(cd.cls)
    idx = greedy_coreset(bank, m, method.projection_dim, seed=seed)
    head = NearestHead(bank[idx].float(), half_bank=method.key == "resnet18").to(config.DEVICE)
    return Fitted(head, mean, std, len(bank), time.perf_counter() - t0)


def run_class(method: config.MethodConfig, model, cls: str) -> dict:
    cd = load_class(cls)
    fitted = fit(method, model, cd)
    out_dir = config.EVAL_DIR / "raw" / method.key
    out_dir.mkdir(parents=True, exist_ok=True)
    mdir = config.MODELS_DIR / method.key
    mdir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "bank": fitted.head.bank.cpu(),
            "norm_mean": fitted.norm_mean,
            "norm_std": fitted.norm_std,
        },
        mdir / f"{cls}.pt",
    )

    val_imgs, _, _ = cd.select("val")
    val_maps, _ = score_embeddings(fitted.head, embed(method, model, val_imgs, fitted))
    val_scores = val_maps.reshape(len(val_maps), -1).max(1)
    thr = float(np.quantile(val_scores, config.THRESHOLD_QUANTILE))

    t0 = time.perf_counter()
    test_imgs, test_masks, test_items = cd.select("test")
    test_maps, _ = score_embeddings(fitted.head, embed(method, model, test_imgs, fitted))
    sec_test = time.perf_counter() - t0
    scores = test_maps.reshape(len(test_maps), -1).max(1)
    labels = np.array([it.label for it in test_items])

    ious = np.full(len(test_items), np.nan)
    area = np.zeros(len(test_items))
    for i, it in enumerate(test_items):
        if it.label:
            ious[i] = metrics.iou(test_maps[i] > thr, test_masks[i])
            area[i] = test_masks[i].mean()
    ph = metrics.pixel_hists(test_maps, test_masks, config.PIXEL_BINS, seed=config.SEED)

    # Pemeriksaan: pixel AUROC dari histogram vs eksak pada seluruh piksel.
    exact_pix = metrics.auroc(test_maps.reshape(-1), test_masks.reshape(-1))
    hist_pix = metrics.auroc_from_hist(ph.pos.sum(0), ph.neg.sum(0))

    np.savez_compressed(
        out_dir / f"{cls}.npz",
        scores=scores.astype(np.float32),
        labels=labels,
        val_scores=val_scores.astype(np.float32),
        threshold=np.float32(thr),
        ious=ious,
        area=area,
        defect=np.array([it.defect for it in test_items]),
        image=np.array([it.image for it in test_items]),
        edges=ph.edges,
        pos=ph.pos.astype(np.int32),
        neg=ph.neg.astype(np.int32),
        region_hist=ph.region_hist.astype(np.int32),
        region_image=ph.region_image,
    )
    summary = {
        "method": method.key,
        "class": cls,
        "fit_images": int(sum(it.split == "fit" for it in cd.items)),
        "val_images": int(len(val_imgs)),
        "test_images": int(len(test_items)),
        "fit_patches": fitted.fit_patches,
        "bank_size": int(fitted.head.bank.shape[0]),
        "seconds_fit": round(fitted.seconds_fit, 1),
        "ms_per_test_image": round(1000 * sec_test / len(test_items), 1),
        "threshold": thr,
        "image_auroc": metrics.auroc(scores, labels),
        "pixel_auroc": exact_pix,
        "pixel_auroc_hist": hist_pix,
        "aupro": metrics.aupro_from_hist(ph.region_hist, ph.neg.sum(0), config.AUPRO_FPR_LIMIT),
        "miou": float(np.nanmean(ious)),
        "test_false_reject": float((scores[labels == 0] > thr).mean()),
        "test_miss": float((scores[labels == 1] <= thr).mean()),
    }
    (out_dir / f"{cls}.json").write_text(json.dumps(summary, indent=2))
    return summary
