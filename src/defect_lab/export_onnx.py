"""Ekspor model browser (PatchCore ResNet18) ke ONNX dan kuantisasi int8.

Dua jenis berkas:
  backbone_int8.onnx  citra [1,3,256,256] 0..1 -> fitur patch [1,1024,384]
                      (normalisasi ImageNet ada di dalam graf; kuantisasi statis QDQ,
                      kalibrasi hanya dari citra normal fit)
  head_<kelas>.onnx   fitur patch -> peta skor [1,256,256] dan skor patch [1,1024]
                      (bank memori float16 + jarak tetangga terdekat + bilinear + Gaussian)

Lalu paritas diuji dengan onnxruntime pada SELURUH citra test: selisih skor terhadap
PyTorch dan perubahan image/pixel AUROC akibat kuantisasi.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
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


def export_all() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    model = PatchEmbedder(METHOD.backbone, METHOD.embed_dim)
    fp32 = config.MODELS_DIR / "resnet18" / "backbone_fp32.onnx"
    export_backbone(model, fp32)
    int8 = OUT / "backbone_int8.onnx"
    quantize_static(
        str(fp32), str(int8), _Calib(calibration_images()),
        quant_format=QuantFormat.QDQ, per_channel=True,
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
    )  # fmt: skip
    onnx.checker.check_model(str(int8))
    s_fp32, s_int8 = session(fp32), session(int8)

    report: dict = {
        "backbone": {
            "file": int8.name,
            "bytes": int8.stat().st_size,
            "fp32_bytes": fp32.stat().st_size,
            "sha256": sha256(int8),
            "opset": OPSET,
            "calibration_images": 8 * len(config.CLASSES),
        },
        "classes": {},
    }
    for cls in config.CLASSES:
        saved = torch.load(config.MODELS_DIR / "resnet18" / f"{cls}.pt", weights_only=True)
        head = NearestHead(saved["bank"].float(), half_bank=True)
        hpath = OUT / f"head_{cls}.onnx"
        export_head(head, hpath)
        s_head = session(hpath)

        cd = load_class(cls)
        imgs, masks, items = cd.select("test")
        labels = np.array([it.label for it in items])
        with torch.no_grad():
            t_maps, _ = score_embeddings(head, embed_images(model, imgs))
        o32, o8 = [], []
        for i in range(len(imgs)):
            x = to_tensor(imgs[i : i + 1]).numpy()
            for sess, out in ((s_fp32, o32), (s_int8, o8)):
                emb = sess.run(None, {"image": x})[0]
                out.append(s_head.run(None, {"patches": emb})[0][0])
        o32, o8 = np.stack(o32), np.stack(o8)
        thr = json.loads((config.EVAL_DIR / "raw" / "resnet18" / f"{cls}.json").read_text())[
            "threshold"
        ]
        sc = {k: v.reshape(len(v), -1).max(1) for k, v in
              (("torch", t_maps), ("onnx_fp32", o32), ("onnx_int8", o8))}  # fmt: skip
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
        }
        print(cls, {k: round(v, 4) if isinstance(v, float) else v
                    for k, v in report["classes"][cls].items() if "sha" not in k})  # fmt: skip
    c = report["classes"].values()
    report["macro"] = {
        k: float(np.mean([x[k] for x in c]))
        for k in ("image_auroc_torch", "image_auroc_int8", "pixel_auroc_torch", "pixel_auroc_int8")
    }
    report["decision_flips_int8_total"] = int(sum(x["decision_flips_int8"] for x in c))
    report["test_images_total"] = int(sum(x["test_images"] for x in c))
    (config.EVAL_DIR / "onnx_parity.json").write_text(json.dumps(report, indent=2))
    return report["macro"]
