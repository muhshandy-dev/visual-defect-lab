"""Pemeriksaan independen atas angka yang dipublikasikan.

Skrip ini sengaja TIDAK memakai modul metrik atau pelaporan dari paket defect_lab. Ia membaca
evaluation/split.csv, evaluation/raw/*, evaluation/results.json, dan data website
(web/public/data), lalu memeriksa:
  1. Pembagian: fit, val, dan test tidak tumpang tindih; test sama dengan split resmi
     (semua citra cacat ada di test; jumlah per kelas cocok); skor test di raw memakai citra
     test yang sama untuk semua metode.
  2. Ambang = kuantil 0,95 skor validasi (dari raw), bukan dari test.
  3. image AUROC (sklearn), pixel AUROC (dari histogram), AUPRO (implementasi ulang
     sederhana), dan mIoU dihitung ulang dan sama dengan results.json dan summary.json.
  4. Skor sampel website sama dengan raw, dan nilai maksimum PNG peta skor sama dengan
     skor citra yang dikodekan.
Keluar dengan kode 1 bila ada yang tidak cocok.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"
WEB = ROOT / "web" / "public"
problems: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        problems.append(msg)


def close(a: float, b: float, tol: float = 1e-6) -> bool:
    return (np.isnan(a) and np.isnan(b)) or abs(a - b) <= tol


def hist_auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    # Peluang skor piksel cacat > skor piksel normal, seri dalam bin dihitung setengah.
    total = 0.0
    below = 0.0
    for p, n in zip(pos.astype(float), neg.astype(float), strict=True):
        total += p * (below + 0.5 * n)
        below += n
    return total / (pos.sum() * neg.sum())


def aupro(region: np.ndarray, neg: np.ndarray, limit: float = 0.3) -> float:
    xs, ys = [0.0], [0.0]
    reg = region.astype(float)
    sizes = reg.sum(1)
    above_r = np.zeros(len(reg))
    above_n = 0.0
    for b in range(reg.shape[1] - 1, -1, -1):
        above_r += reg[:, b]
        above_n += neg[b]
        xs.append(above_n / neg.sum())
        ys.append(float(np.mean(above_r / sizes)))
    xs, ys = np.array(xs), np.array(ys)
    grid_y = np.interp(limit, xs, ys)
    keep = xs <= limit
    x = np.append(xs[keep], limit)
    y = np.append(ys[keep], grid_y)
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2) / limit)


# 1. Pembagian
rows = list(csv.DictReader((EVAL / "split.csv").open()))
by_split: dict[str, set[str]] = {}
for r in rows:
    by_split.setdefault(r["split"], set()).add(r["image"])
check(not (by_split["fit"] & by_split["val"]), "fit dan val tumpang tindih")
check(
    not (by_split["test"] & (by_split["fit"] | by_split["val"])),
    "test tumpang tindih dengan fit/val",
)
check(all(r["split"] == "test" for r in rows if r["label"] == "1"), "ada citra cacat di luar test")
check(len(rows) == 10821, f"jumlah citra {len(rows)}, VisA 1cls = 10821")
test_rows = {r["class"]: [] for r in rows}
for r in rows:
    if r["split"] == "test":
        test_rows[r["class"]].append(r)
cnt = Counter((r["class"], r["split"]) for r in rows)
for cls in test_rows:
    n_train = cnt[(cls, "fit")] + cnt[(cls, "val")]
    check(cnt[(cls, "val")] == round(n_train * 0.1), f"{cls}: val bukan 10% dari train")

results = json.loads((EVAL / "results.json").read_text())
summary = json.loads((WEB / "data" / "summary.json").read_text())
methods = summary["method_order"]
check(set(methods) <= set(results["methods"]), "metode website tidak ada di results.json")

for m in methods:
    macro = {k: [] for k in ("image_auroc", "pixel_auroc", "aupro", "miou")}
    for cls, trows in test_rows.items():
        z = np.load(EVAL / "raw" / m / f"{cls}.npz")
        meta = json.loads((EVAL / "raw" / m / f"{cls}.json").read_text())
        images = z["image"].tolist()
        check(
            images == sorted(r["image"] for r in trows),
            f"{m}/{cls}: citra test tidak sama dengan split",
        )
        labels = z["labels"]
        want = [int(r["label"]) for r in sorted(trows, key=lambda r: r["image"])]
        check(labels.tolist() == want, f"{m}/{cls}: label test berbeda")
        # 2. Ambang dari validasi
        thr = float(np.quantile(z["val_scores"].astype(np.float64), 0.95))
        check(close(thr, float(z["threshold"]), 1e-5), f"{m}/{cls}: ambang bukan kuantil val")
        check(len(z["val_scores"]) == cnt[(cls, "val")], f"{m}/{cls}: jumlah skor val")
        # 3. Metrik
        ia = roc_auc_score(labels, z["scores"])
        pa = hist_auroc(z["pos"].sum(0), z["neg"].sum(0))
        ap = aupro(z["region_hist"], z["neg"].sum(0))
        mi = float(np.nanmean(z["ious"][labels == 1]))
        check(
            abs(pa - meta["pixel_auroc"]) < 2e-3,
            f"{m}/{cls}: pixel AUROC histogram jauh dari eksak",
        )
        ref = results["methods"][m]["per_class"][cls]
        web = summary["methods"][m]["per_class"][cls]
        for k, v in (("image_auroc", ia), ("pixel_auroc", pa), ("aupro", ap), ("miou", mi)):
            check(close(v, ref[k], 1e-6), f"{m}/{cls}: {k} {v} vs results {ref[k]}")
            check(close(v, web[k], 1e-6), f"{m}/{cls}: {k} {v} vs website {web[k]}")
            macro[k].append(v)
        fr = float((z["scores"][labels == 0] > thr).mean())
        check(close(fr, web["test_false_reject"], 1e-6), f"{m}/{cls}: tingkat tolak normal")
        # 4. Sampel website
        samples = json.loads((WEB / "data" / "samples" / f"{cls}.json").read_text())
        check(
            close(samples["thresholds"][m], float(z["threshold"]), 1e-6),
            f"{m}/{cls}: ambang website",
        )
        for s in samples["items"]:
            cand = [j for j, p in enumerate(images) if Path(p).stem == s["id"][1:]
                    and (("Anomaly" in p) == (s["id"][0] == "a"))]  # fmt: skip
            check(len(cand) == 1, f"{cls}/{s['id']}: tidak ditemukan di test")
            if len(cand) != 1:
                continue
            i = cand[0]
            check(close(s["scores"][m], float(z["scores"][i]), 1e-6), f"{m}/{cls}/{s['id']}: skor")
            check(s["label"] == int(labels[i]), f"{cls}/{s['id']}: label")
            q = np.asarray(Image.open(WEB / s["maps"][m]))
            r = min(max(float(z["scores"][i]) / float(z["threshold"]), 0), 2)
            want_v = int(np.floor(r / 2 * 255 + 0.5))
            check(
                abs(int(q.max()) - want_v) <= 1,
                f"{m}/{cls}/{s['id']}: maks PNG {q.max()} vs {want_v}",
            )
    for k, vals in macro.items():
        check(
            close(float(np.mean(vals)), summary["methods"][m]["macro"][k], 1e-6), f"{m}: makro {k}"
        )

if problems:
    print(f"{len(problems)} masalah:")
    print("\n".join(problems[:50]))
    sys.exit(1)
print(f"OK: {len(methods)} metode x {len(test_rows)} kelas diperiksa ulang secara independen")
