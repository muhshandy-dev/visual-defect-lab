"""Gabungkan keluaran mentah semua metode: rata-rata makro, bootstrap 95%, perbandingan
berpasangan, analisis menurut ukuran cacat dan jenis cacat. Menulis evaluation/results.json.

Bootstrap: dalam tiap kelas, citra normal dan citra cacat test diambil ulang dengan
pengembalian (stratifikasi). Indeks ulangan bergantung hanya pada (benih, kelas, ulangan),
jadi semua metode memakai sampel ulang yang sama dan selisihnya berpasangan.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from . import config, metrics

METRICS = ("image_auroc", "pixel_auroc", "aupro", "miou")
# Laju kesalahan pada ambang operasi (q0,95 skor validasi normal), ikut dibootstrap
# dengan sampel ulang yang sama sehingga punya CI 95% dan selisih berpasangan.
OP_METRICS = ("false_reject", "miss")
BOOT_KEYS = METRICS + OP_METRICS
# Batas kelompok ukuran cacat (bagian luas citra), ditetapkan sebelum melihat skor.
SIZE_BINS = ((0.0, 0.002, "kecil (< 0,2%)"), (0.002, 0.01, "sedang (0,2 sampai 1%)"),
             (0.01, 1.01, "besar (> 1%)"))  # fmt: skip


@dataclass
class Raw:
    scores: np.ndarray
    labels: np.ndarray
    threshold: float
    ious: np.ndarray
    area: np.ndarray
    defect: np.ndarray
    image: np.ndarray
    pos: np.ndarray
    neg: np.ndarray
    region_hist: np.ndarray
    region_image: np.ndarray


def load_raw(method: str, cls: str) -> Raw:
    z = np.load(config.EVAL_DIR / "raw" / method / f"{cls}.npz")
    return Raw(
        z["scores"].astype(np.float64), z["labels"], float(z["threshold"]), z["ious"],
        z["area"], z["defect"], z["image"], z["pos"], z["neg"], z["region_hist"],
        z["region_image"],
    )  # fmt: skip


def class_metrics(r: Raw, idx: np.ndarray) -> dict[str, float]:
    """Metrik pada subset citra `idx` (boleh berulang, untuk bootstrap)."""
    counts = np.bincount(idx, minlength=len(r.labels))
    pos = (r.pos * counts[:, None]).sum(0)
    neg = (r.neg * counts[:, None]).sum(0)
    reg_w = counts[r.region_image]
    keep = reg_w > 0
    region = np.repeat(r.region_hist[keep], reg_w[keep], axis=0)
    an = idx[r.labels[idx] == 1]
    nm = idx[r.labels[idx] == 0]
    return {
        "false_reject": float((r.scores[nm] > r.threshold).mean()) if len(nm) else float("nan"),
        "miss": float((r.scores[an] <= r.threshold).mean()) if len(an) else float("nan"),
        "image_auroc": metrics.auroc(r.scores[idx], r.labels[idx]),
        "pixel_auroc": metrics.auroc_from_hist(pos, neg),
        "aupro": metrics.aupro_from_hist(region, neg, config.AUPRO_FPR_LIMIT),
        "miou": float(np.nanmean(r.ious[an])) if len(an) else float("nan"),
    }


def resample(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    parts = []
    for lab in (0, 1):
        ids = np.flatnonzero(labels == lab)
        parts.append(rng.choice(ids, size=len(ids), replace=True))
    return np.concatenate(parts)


def build_results(methods: list[str], reps: int = config.BOOTSTRAP_REPS) -> dict:
    raws = {m: {c: load_raw(m, c) for c in config.CLASSES} for m in methods}
    summaries = {
        m: {
            c: json.loads((config.EVAL_DIR / "raw" / m / f"{c}.json").read_text())
            for c in config.CLASSES
        }
        for m in methods
    }
    # Semua metode harus menilai citra test yang sama dengan urutan yang sama.
    for c in config.CLASSES:
        ref = raws[methods[0]][c].image
        for m in methods[1:]:
            assert np.array_equal(raws[m][c].image, ref), f"urutan test berbeda: {m}/{c}"

    point = {m: {c: class_metrics(raws[m][c], np.arange(len(raws[m][c].labels)))
                 for c in config.CLASSES} for m in methods}  # fmt: skip
    boot = {m: {k: np.zeros((reps, len(config.CLASSES))) for k in BOOT_KEYS} for m in methods}
    for ci, c in enumerate(config.CLASSES):
        labels = raws[methods[0]][c].labels
        rng = np.random.default_rng([config.SEED, ci, 7])
        for b in range(reps):
            idx = resample(labels, rng)
            for m in methods:
                vals = class_metrics(raws[m][c], idx)
                for k in BOOT_KEYS:
                    boot[m][k][b, ci] = vals[k]

    out: dict = {
        "config": {
            "image_size": config.IMAGE_SIZE,
            "threshold_quantile": config.THRESHOLD_QUANTILE,
            "smooth_sigma": config.SMOOTH_SIGMA,
            "bootstrap_reps": reps,
            "pixel_bins": config.PIXEL_BINS,
            "aupro_fpr_limit": config.AUPRO_FPR_LIMIT,
            "seed": config.SEED,
            "methods": {m: config.METHODS[m].__dict__ for m in methods},
        },
        "methods": {},
        "paired": {},
    }
    for m in methods:
        per_class = {}
        for ci, c in enumerate(config.CLASSES):
            s = summaries[m][c]
            r = raws[m][c]
            entry = {k: point[m][c][k] for k in METRICS}
            entry["ci95"] = {k: metrics.bootstrap_ci(boot[m][k][:, ci]) for k in BOOT_KEYS}
            n_def = int((r.labels == 1).sum())
            n_norm = int((r.labels == 0).sum())
            entry.update({
                "test_defect_images": n_def,
                "test_normal_images": n_norm,
                "test_missed": int(round(point[m][c]["miss"] * n_def)),
                "test_false_rejected": int(round(point[m][c]["false_reject"] * n_norm)),
                "threshold": r.threshold,
                "test_false_reject": s["test_false_reject"],
                "test_miss": s["test_miss"],
                "pixel_auroc_exact": s["pixel_auroc"],
                "fit_images": s["fit_images"], "val_images": s["val_images"],
                "test_images": s["test_images"], "bank_size": s["bank_size"],
                "fit_patches": s["fit_patches"], "seconds_fit": s["seconds_fit"],
                "ms_per_test_image": s["ms_per_test_image"],
            })  # fmt: skip
            per_class[c] = entry
        macro = {k: float(np.mean([point[m][c][k] for c in config.CLASSES])) for k in METRICS}
        macro_ci = {k: metrics.bootstrap_ci(boot[m][k].mean(1)) for k in BOOT_KEYS}
        fr = float(np.mean([summaries[m][c]["test_false_reject"] for c in config.CLASSES]))
        miss = float(np.mean([summaries[m][c]["test_miss"] for c in config.CLASSES]))
        out["methods"][m] = {
            "label": config.METHODS[m].label,
            "macro": macro,
            "macro_ci95": macro_ci,
            "macro_false_reject": fr,
            "macro_miss": miss,
            "per_class": per_class,
            "by_size": size_breakdown(raws[m]),
            "by_defect": defect_breakdown(raws[m]),
        }
    base = "classical"
    for m in methods:
        if m == base:
            continue
        diff = {}
        for k in BOOT_KEYS:
            d = boot[m][k].mean(1) - boot[base][k].mean(1)
            point_d = macro_of(out, m, k) - macro_of(out, base, k)
            diff[k] = {"diff": point_d, "ci95": metrics.bootstrap_ci(d),
                       "p_le_zero": float((d <= 0).mean())}  # fmt: skip
        out["paired"][f"{m}_vs_{base}"] = diff
    if "wrn50" in methods and "resnet18" in methods:
        diff = {}
        for k in BOOT_KEYS:
            d = boot["wrn50"][k].mean(1) - boot["resnet18"][k].mean(1)
            point_d = macro_of(out, "wrn50", k) - macro_of(out, "resnet18", k)
            diff[k] = {"diff": point_d, "ci95": metrics.bootstrap_ci(d),
                       "p_le_zero": float((d <= 0).mean())}  # fmt: skip
        out["paired"]["wrn50_vs_resnet18"] = diff
    return out


def macro_of(out: dict, m: str, k: str) -> float:
    e = out["methods"][m]
    return e["macro"][k] if k in e["macro"] else e[f"macro_{k}"]


def size_breakdown(raws: dict[str, Raw]) -> list[dict]:
    rows = []
    for lo, hi, name in SIZE_BINS:
        det, ious, n = [], [], 0
        for r in raws.values():
            sel = (r.labels == 1) & (r.area >= lo) & (r.area < hi)
            n += int(sel.sum())
            det.extend((r.scores[sel] > r.threshold).tolist())
            ious.extend(r.ious[sel].tolist())
        rows.append({
            "group": name, "n": n,
            "detected": float(np.mean(det)) if det else float("nan"),
            "miou": float(np.nanmean(ious)) if ious else float("nan"),
        })  # fmt: skip
    return rows


def defect_breakdown(raws: dict[str, Raw]) -> dict[str, list[dict]]:
    out = {}
    for c, r in raws.items():
        rows = []
        for d in sorted({x for x in r.defect.tolist() if x}):
            sel = (r.labels == 1) & (r.defect == d)
            rows.append({
                "defect": d, "n": int(sel.sum()),
                "detected": float((r.scores[sel] > r.threshold).mean()),
                "miou": float(np.nanmean(r.ious[sel])),
            })  # fmt: skip
        out[c] = rows
    return out
