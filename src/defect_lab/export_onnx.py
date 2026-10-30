"""Ekspor model browser (PatchCore ResNet18) ke ONNX dan kuantisasi int8.

Dua jenis berkas:
  backbone_int8.onnx  citra [1,3,256,256] 0..1 -> fitur patch [1,1024,384]
                      (normalisasi ImageNet ada di dalam graf; bobot konvolusi int8)
  head_<kelas>.onnx   fitur patch -> peta skor [1,256,256] dan skor patch [1,1024]
                      (bank memori float16 + jarak tetangga terdekat + bilinear + Gaussian)

Dua cara kuantisasi dibandingkan:
  weights  bobot konvolusi int8 per kanal (simetris), aktivasi tetap float32
  static   kuantisasi statis QDQ bobot dan aktivasi (kalibrasi dari citra normal fit)
Varian yang dipakai website dipilih dengan aturan yang ditetapkan sebelum melihat data uji:
kesetiaan terhadap PyTorch pada citra normal VALIDASI, yaitu selisih skor maksimum dibagi
ambang. Varian terkecil dengan selisih <= FIDELITY_LIMIT dipilih; bila tidak ada, varian
dengan selisih terkecil dipakai dan laporan menandainya gagal. Label uji tidak dipakai.

Lalu paritas diuji dengan onnxruntime pada SELURUH citra test untuk semua varian: selisih
skor terhadap PyTorch dan perubahan image/pixel AUROC akibat kuantisasi.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import helper, numpy_helper
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

from . import config, metrics
from .data import load_class
from .patchcore import NearestHead, PatchEmbedder, embed_images, score_embeddings, to_tensor

METHOD = config.METHODS["resnet18"]
OUT = config.WEB_PUBLIC / "models"
OPSET = 17
FIDELITY_LIMIT = 0.05  # selisih skor maksimum / ambang pada citra normal validasi
VARIANTS = ("weights", "static")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_backbone(model: PatchEmbedder, path: Path) -> None:
    dummy = torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    torch.onnx.export(
        model, dummy, str(path), input_names=["image"], output_names=["patches"],
        opset_version=OPSET, dynamo=False,
    )  # fmt: skip


def export_head(head: NearestHead, path: Path) -> None:
    # Tanpa constant folding agar bank tetap float16 (Cast dijalankan saat inferensi).
    dummy = torch.zeros(1, 1024, METHOD.embed_dim)
    torch.onnx.export(
        head, dummy, str(path), input_names=["patches"], output_names=["map", "patch_scores"],
        opset_version=OPSET, dynamo=False, do_constant_folding=False,
    )  # fmt: skip


class _Calib(CalibrationDataReader):
    def __init__(self, images: np.ndarray):
        self.it = iter(to_tensor(images[i : i + 1]).numpy() for i in range(len(images)))

    def get_next(self):
        x = next(self.it, None)
        return None if x is None else {"image": x}


def calibration_images(per_class: int = 8) -> np.ndarray:
    imgs = []
    for ci, cls in enumerate(config.CLASSES):
        fit_imgs, _, _ = load_class(cls).select("fit")
        rng = np.random.default_rng([config.SEED, ci, 11])
        imgs.append(fit_imgs[rng.choice(len(fit_imgs), per_class, replace=False)])
    return np.concatenate(imgs)


def session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def quantize_weights(src: Path, dst: Path) -> None:
    """Bobot Conv menjadi int8 per kanal keluaran (simetris, titik nol 0) + DequantizeLinear.

    Aktivasi tetap float32, jadi galat hanya dari pembulatan bobot (paling besar setengah
    langkah kuantisasi per kanal). Berkas kira-kira seperempat ukuran float32.
    """
    m = onnx.load(str(src))
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    weights = sorted({n.input[1] for n in g.node if n.op_type == "Conv" and n.input[1] in inits})
    dq_nodes = []
    for name in weights:
        w = numpy_helper.to_array(inits[name]).astype(np.float32)
        absmax = np.abs(w).reshape(w.shape[0], -1).max(1)
        scale = np.where(absmax > 0, absmax / 127.0, 1.0).astype(np.float32)
        q = np.clip(np.round(w / scale.reshape(-1, *[1] * (w.ndim - 1))), -127, 127)
        g.initializer.remove(inits[name])
        g.initializer.extend([
            numpy_helper.from_array(q.astype(np.int8), f"{name}_q"),
            numpy_helper.from_array(scale, f"{name}_scale"),
            numpy_helper.from_array(np.zeros(len(scale), np.int8), f"{name}_zp"),
        ])  # fmt: skip
        dq_nodes.append(helper.make_node(
            "DequantizeLinear", [f"{name}_q", f"{name}_scale", f"{name}_zp"], [name], axis=0,
        ))  # fmt: skip
    for i, node in enumerate(dq_nodes):
        g.node.insert(i, node)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))


def quantize_activations(src: Path, dst: Path) -> None:
    quantize_static(
        str(src), str(dst), _Calib(calibration_images()),
        quant_format=QuantFormat.QDQ, per_channel=True,
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
    )  # fmt: skip
    onnx.checker.check_model(str(dst))


def browser_scores(s_bb: ort.InferenceSession, s_head: ort.InferenceSession,
                   imgs: np.ndarray) -> np.ndarray:  # fmt: skip
    out = []
    for i in range(len(imgs)):
        emb = s_bb.run(None, {"image": to_tensor(imgs[i : i + 1]).numpy()})[0]
        out.append(s_head.run(None, {"patches": emb})[0][0])
    return np.stack(out)


def export_all() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    work = config.MODELS_DIR / "resnet18"
    model = PatchEmbedder(METHOD.backbone, METHOD.embed_dim)
    fp32 = work / "backbone_fp32.onnx"
    export_backbone(model, fp32)
    files = {"weights": work / "backbone_weights_int8.onnx",
             "static": work / "backbone_static_int8.onnx"}  # fmt: skip
    quantize_weights(fp32, files["weights"])
    quantize_activations(fp32, files["static"])
    s_fp32 = session(fp32)
    s_var = {v: session(files[v]) for v in VARIANTS}

    heads, thresholds = {}, {}
    for cls in config.CLASSES:
        saved = torch.load(work / f"{cls}.pt", weights_only=True)
        heads[cls] = NearestHead(saved["bank"].float(), half_bank=True)
        hpath = OUT / f"head_{cls}.onnx"
        export_head(heads[cls], hpath)
        thresholds[cls] = json.loads(
            (config.EVAL_DIR / "raw" / "resnet18" / f"{cls}.json").read_text()
        )["threshold"]

    # Pemilihan varian: hanya citra normal validasi (tanpa label uji).
    fidelity = {v: [] for v in VARIANTS}
    for cls in config.CLASSES:
        s_head = session(OUT / f"head_{cls}.onnx")
        val_imgs, _, _ = load_class(cls).select("val")
        with torch.no_grad():
            ref, _ = score_embeddings(heads[cls], embed_images(model, val_imgs))
        ref = ref.reshape(len(ref), -1).max(1)
        for v in VARIANTS:
            got = browser_scores(s_var[v], s_head, val_imgs)
            got = got.reshape(len(got), -1).max(1)
            fidelity[v].append(float(np.abs(got - ref).max() / thresholds[cls]))
    worst = {v: max(fidelity[v]) for v in VARIANTS}
    sizes = {v: files[v].stat().st_size for v in VARIANTS}
    passing = [v for v in VARIANTS if worst[v] <= FIDELITY_LIMIT]
    chosen = min(passing, key=sizes.get) if passing else min(VARIANTS, key=worst.get)
    int8 = OUT / "backbone_int8.onnx"
    int8.write_bytes(files[chosen].read_bytes())
    print("kesetiaan validasi (selisih maks / ambang):", worst, "dipilih:", chosen)

    report: dict = {
        "backbone": {
            "file": int8.name,
            "variant": chosen,
            "bytes": int8.stat().st_size,
            "fp32_bytes": fp32.stat().st_size,
            "sha256": sha256(int8),
            "opset": OPSET,
            "calibration_images": 8 * len(config.CLASSES),
        },
        "selection": {
            "rule": "varian terkecil dengan selisih skor maks / ambang <= "
            f"{FIDELITY_LIMIT} pada citra normal validasi",
            "fidelity_limit": FIDELITY_LIMIT,
            "val_max_rel_diff": worst,
            "val_max_rel_diff_per_class": {
                v: dict(zip(config.CLASSES, fidelity[v], strict=True)) for v in VARIANTS
            },
            "bytes": sizes,
            "passed": bool(passing),
        },
        "classes": {},
    }
    for cls in config.CLASSES:
        head = heads[cls]
        hpath = OUT / f"head_{cls}.onnx"
        s_head = session(hpath)

        cd = load_class(cls)
        imgs, masks, items = cd.select("test")
        labels = np.array([it.label for it in items])
        with torch.no_grad():
            t_maps, _ = score_embeddings(head, embed_images(model, imgs))
        o32 = browser_scores(s_fp32, s_head, imgs)
        ov = {v: browser_scores(s_var[v], s_head, imgs) for v in VARIANTS}
        o8 = ov[chosen]
        thr = thresholds[cls]
        sc = {k: v.reshape(len(v), -1).max(1) for k, v in
              (("torch", t_maps), ("onnx_fp32", o32), ("onnx_int8", o8))}  # fmt: skip
        other = {v: ov[v].reshape(len(ov[v]), -1).max(1) for v in VARIANTS}
        report["classes"][cls] = {
            "head_bytes": hpath.stat().st_size,
            "head_sha256": sha256(hpath),
            "bank_size": int(head.bank.shape[0]),
            "max_abs_score_diff_fp32": float(np.abs(sc["onnx_fp32"] - sc["torch"]).max()),
            "max_abs_score_diff_int8": float(np.abs(sc["onnx_int8"] - sc["torch"]).max()),
            "threshold": thr,
            "decision_flips_int8": int(((sc["onnx_int8"] > thr) != (sc["torch"] > thr)).sum()),
            "image_auroc_torch": metrics.auroc(sc["torch"], labels),
            "image_auroc_int8": metrics.auroc(sc["onnx_int8"], labels),
            "pixel_auroc_torch": metrics.auroc(t_maps.reshape(-1), masks.reshape(-1)),
            "pixel_auroc_int8": metrics.auroc(o8.reshape(-1), masks.reshape(-1)),
            "test_images": len(imgs),
            "variants": {
                v: {
                    "max_abs_score_diff": float(np.abs(other[v] - sc["torch"]).max()),
                    "decision_flips": int(((other[v] > thr) != (sc["torch"] > thr)).sum()),
                    "image_auroc": metrics.auroc(other[v], labels),
                }
                for v in VARIANTS
            },
        }
        print(cls, {k: round(v, 4) if isinstance(v, float) else v
                    for k, v in report["classes"][cls].items()
                    if "sha" not in k and k != "variants"})  # fmt: skip
    c = report["classes"].values()
    report["macro"] = {
        k: float(np.mean([x[k] for x in c]))
        for k in ("image_auroc_torch", "image_auroc_int8", "pixel_auroc_torch", "pixel_auroc_int8")
    }
    report["decision_flips_int8_total"] = int(sum(x["decision_flips_int8"] for x in c))
    report["test_images_total"] = int(sum(x["test_images"] for x in c))
    (config.EVAL_DIR / "onnx_parity.json").write_text(json.dumps(report, indent=2))
    return report["macro"]
