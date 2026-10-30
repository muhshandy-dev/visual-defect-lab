"""Tulis data website ke web/public: ringkasan, sampel per produk, foto, masker, peta skor.

Peta skor sampel dihitung ulang dari bank memori yang tersimpan dan dicek sama dengan skor
pada evaluation/raw (selisih maksimum dicatat; ekspor gagal bila melebihi toleransi).
Peta disimpan sebagai PNG abu-abu: v = floor(min(skor/ambang, 2) / 2 * 255 + 0,5),
sama dengan encodeRatio di web/src/lib/heat.ts.
"""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import torch

from . import config
from .data import load_class
from .evaluate import Fitted, embed, load_embedder
from .patchcore import NearestHead, score_embeddings
from .report import load_raw

ORDER = ("wrn50", "resnet18", "classical")
N_RANDOM_DEFECT = 5
N_RANDOM_NORMAL = 3
N_FAIL = 2
DISPLAY_LONG_SIDE = 720
TOL = 1e-3

# Sama dengan STOPS di web/src/lib/heat.ts.
STOPS = [
    (0.55, (120, 53, 15), 0.0),
    (0.8, (180, 83, 9), 0.45),
    (1.0, (245, 158, 11), 0.72),
    (1.4, (252, 211, 77), 0.85),
    (2.0, (254, 243, 199), 0.92),
]


def encode(score_map: np.ndarray, thr: float) -> np.ndarray:
    r = np.clip(score_map / thr, 0, 2)
    return np.floor(r / 2 * 255 + 0.5).astype(np.uint8)


def heat_overlay(rgb: np.ndarray, q: np.ndarray) -> np.ndarray:
    ratio = q.astype(np.float64) / 255 * 2
    color = np.zeros(q.shape + (3,))
    alpha = np.zeros(q.shape)
    for (r0, c0, a0), (r1, c1, a1) in zip(STOPS[:-1], STOPS[1:], strict=True):
        sel = (ratio > r0) & (ratio <= r1)
        t = ((ratio - r0) / (r1 - r0))[sel]
        color[sel] = np.array(c0) + t[:, None] * (np.array(c1) - np.array(c0))
        alpha[sel] = a0 + t * (a1 - a0)
    top = ratio > STOPS[-1][0]
    color[top], alpha[top] = STOPS[-1][1], STOPS[-1][2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)[..., None] * 0.55
    out = gray * (1 - alpha[..., None]) + color * alpha[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def sample_id(image_path: str) -> str:
    p = Path(image_path)
    return ("a" if "Anomaly" in p.parts else "n") + p.stem


def pick(raw, ci: int) -> list[tuple[int, str]]:
    rng = np.random.default_rng([config.SEED, ci, 3])
    ratio = raw.scores / raw.threshold
    defect = np.flatnonzero(raw.labels == 1)
    normal = np.flatnonzero(raw.labels == 0)
    chosen = [(int(i), "acak") for i in rng.choice(defect, N_RANDOM_DEFECT, replace=False)]
    chosen += [(int(i), "acak") for i in rng.choice(normal, N_RANDOM_NORMAL, replace=False)]
    taken = {i for i, _ in chosen}
    missed = [i for i in defect[np.argsort(ratio[defect])] if ratio[i] <= 1 and i not in taken]
    false = [i for i in normal[np.argsort(-ratio[normal])] if ratio[i] > 1 and i not in taken]
    chosen += [(int(i), "terlewat") for i in missed[:N_FAIL]]
    chosen += [(int(i), "alarm-palsu") for i in false[:N_FAIL]]
    return chosen


def load_fitted(method: config.MethodConfig, cls: str) -> Fitted:
    saved = torch.load(config.MODELS_DIR / method.key / f"{cls}.pt", weights_only=True)
    head = NearestHead(saved["bank"].float(), half_bank=method.key == "resnet18")
    return Fitted(head, saved["norm_mean"], saved["norm_std"], 0, 0.0)


def write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), arr, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
        raise OSError(path)


def write_jpg(path: Path, rgb: np.ndarray, quality: int = 82) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise OSError(path)


def int8_scores(cls: str, inputs: np.ndarray) -> list[float] | None:
    """Skor model browser (ONNX int8) dihitung onnxruntime Python, untuk uji paritas web."""
    bb = config.WEB_PUBLIC / "models" / "backbone_int8.onnx"
    hd = config.WEB_PUBLIC / "models" / f"head_{cls}.onnx"
    if not (bb.exists() and hd.exists()):
        return None
    import onnxruntime as ort

    from .patchcore import to_tensor

    sb = ort.InferenceSession(str(bb), providers=["CPUExecutionProvider"])
    sh = ort.InferenceSession(str(hd), providers=["CPUExecutionProvider"])
    out = []
    for i in range(len(inputs)):
        emb = sb.run(None, {"image": to_tensor(inputs[i : i + 1]).numpy()})[0]
        out.append(float(sh.run(None, {"patches": emb})[0].max()))
    return out


def refresh_summary() -> None:
    """Perbarui angka di summary.json dari evaluation/results.json tanpa menyentuh gambar.

    Dipakai bila laporan dihitung ulang (misalnya CI tambahan) sementara peta skor dan
    sampel tetap sama. Metode dan urutan sampel tidak boleh berubah.
    """
    path = config.WEB_PUBLIC / "data" / "summary.json"
    summary = json.loads(path.read_text())
    results = json.loads((config.EVAL_DIR / "results.json").read_text())
    methods = summary["method_order"]
    assert set(methods) <= set(results["methods"]), "metode di website tidak ada di results.json"
    summary["config"] = results["config"]
    summary["methods"] = {m: {"key": m, **results["methods"][m]} for m in methods}
    summary["paired"] = results["paired"]
    path.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print("summary.json diperbarui dari results.json")


def export() -> None:
    methods = [
        m for m in ORDER if (config.EVAL_DIR / "raw" / m / f"{config.CLASSES[-1]}.npz").exists()
    ]
    if not methods:
        raise SystemExit("belum ada hasil evaluasi")
    results = json.loads((config.EVAL_DIR / "results.json").read_text())
    primary = methods[0]
    pub = config.WEB_PUBLIC
    for sub in ("data", "img", "in", "mask", "map"):
        shutil.rmtree(pub / sub, ignore_errors=True)
    (pub / "data" / "samples").mkdir(parents=True)
    root = config.RAW_DIR / "visa"

    embedders = {m: load_embedder(config.METHODS[m]) for m in methods}
    max_diff = {m: 0.0 for m in methods}
    has_int8 = (pub / "models" / "backbone_int8.onnx").exists()

    for ci, cls in enumerate(config.CLASSES):
        raws = {m: load_raw(m, cls) for m in methods}
        cd = load_class(cls)
        imgs, masks, items = cd.select("test")
        assert [it.image for it in items] == raws[primary].image.tolist()
        chosen = pick(raws[primary], ci)
        idx = np.array([i for i, _ in chosen])
        maps = {}
        for m in methods:
            mc = config.METHODS[m]
            fitted = load_fitted(mc, cls)
            mp, _ = score_embeddings(fitted.head, embed(mc, embedders[m], imgs[idx], fitted))
            sc = mp.reshape(len(mp), -1).max(1)
            diff = float(np.abs(sc - raws[m].scores[idx]).max() / raws[m].threshold)
            max_diff[m] = max(max_diff[m], diff)
            if diff > TOL:
                raise RuntimeError(f"{m}/{cls}: skor ekspor berbeda dari evaluasi ({diff:.2e})")
            maps[m] = mp
        i8 = int8_scores(cls, imgs[idx]) if has_int8 and "resnet18" in methods else None

        samples = []
        for k, (i, group) in enumerate(chosen):
            it = items[i]
            sid = sample_id(it.image)
            full = cv2.cvtColor(cv2.imread(str(root / it.image)), cv2.COLOR_BGR2RGB)
            h, w = full.shape[:2]
            s = DISPLAY_LONG_SIDE / max(h, w)
            disp = cv2.resize(full, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            write_jpg(pub / "img" / cls / f"{sid}.jpg", disp)
            write_png(pub / "in" / cls / f"{sid}.png", imgs[i])
            entry = {
                "id": sid,
                "group": group,
                "label": int(it.label),
                "defect": it.defect,
                "area": float(masks[i].mean()),
                "img": f"img/{cls}/{sid}.jpg",
                "w": int(w),
                "h": int(h),
                "input": f"in/{cls}/{sid}.png",
                "mask": None,
                "maps": {},
                "scores": {},
                "iou": {},
            }
            if it.label:
                write_png(pub / "mask" / cls / f"{sid}.png", masks[i] * 255)
                entry["mask"] = f"mask/{cls}/{sid}.png"
            for m in methods:
                thr = raws[m].threshold
                q = encode(maps[m][k], thr)
                write_png(pub / "map" / m / cls / f"{sid}.png", q)
                entry["maps"][m] = f"map/{m}/{cls}/{sid}.png"
                entry["scores"][m] = float(raws[m].scores[i])
                entry["iou"][m] = None if not it.label else float(raws[m].ious[i])
                if m == primary:
                    write_jpg(
                        pub / "map" / f"{m}_thumb" / cls / f"{sid}.jpg",
                        heat_overlay(imgs[i], q),
                        85,
                    )
                    entry["maps"][f"{m}_thumb"] = f"map/{m}_thumb/{cls}/{sid}.jpg"
            if i8 is not None:
                entry["scores"]["resnet18_int8"] = i8[k]
            samples.append(entry)
        payload = {
            "cls": cls,
            "thresholds": {m: raws[m].threshold for m in methods},
            "items": samples,
        }
        (pub / "data" / "samples" / f"{cls}.json").write_text(json.dumps(payload, indent=1))
        print(cls, len(samples), "sampel", flush=True)

    counts: dict[str, int] = {"fit": 0, "val": 0, "test_normal": 0, "test_defect": 0}
    for cls in config.CLASSES:
        for it in load_class(cls).items:
            key = it.split if it.split != "test" else ("test_defect" if it.label else "test_normal")
            counts[key] += 1

    onnx_report = None
    parity_path = config.EVAL_DIR / "onnx_parity.json"
    if parity_path.exists() and has_int8:
        p = json.loads(parity_path.read_text())
        heads = [c["head_bytes"] for c in p["classes"].values()]
        onnx_report = {
            "backbone": p["backbone"],
            "head_bytes_total": int(sum(heads)),
            "head_bytes_max": int(max(heads)),
            "macro": p["macro"],
            "decision_flips_int8_total": p["decision_flips_int8_total"],
            "test_images_total": p["test_images_total"],
        }

    summary = {
        "generated": date.today().isoformat(),
        "primary": primary,
        "live_method": "resnet18" if onnx_report else None,
        "dataset": {
            "name": "VisA",
            "version": "VisA_20220922",
            "sha256": config.VISA_SHA256,
            "licence": "CC BY 4.0",
            "licence_url": "https://creativecommons.org/licenses/by/4.0/",
            "source_url": "https://github.com/amazon-science/spot-diff",
            "citation": "Zou, Jeong, Pemula, Zhang, Dabeer. SPot-the-Difference Self-Supervised "
            "Pre-training for Anomaly Detection and Segmentation. ECCV 2022.",
            "counts": counts,
        },
        "config": results["config"],
        "method_order": methods,
        "methods": {m: {"key": m, **results["methods"][m]} for m in methods},
        "paired": results["paired"],
        "classes": list(config.CLASSES),
        "onnx": onnx_report,
        "export_max_score_diff": max_diff,
    }
    (pub / "data" / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print("ekspor selesai; selisih skor maksimum (relatif ambang):", max_diff)
