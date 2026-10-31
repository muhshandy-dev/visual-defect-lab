"""Fixture uji jalur inferensi browser dengan bobot ACAK. Bukan model, bukan hasil.

Menyalin web/dist ke folder tujuan, lalu menambahkan backbone ResNet18 berbobot acak
(bobot int8 seperti model asli) dan kepala dengan bank acak untuk beberapa produk.
Skor referensi dihitung dengan onnxruntime Python, sehingga `npm run e2e` dengan
DIST_DIR=<tujuan> menguji: pemuatan onnxruntime-web di bawah CSP, praproses piksel,
eksekusi int8 di WASM, dan paritas skor browser vs Python. Angka dari fixture ini tidak
bermakna dan tidak boleh dipublikasikan.

Pemakaian: python tools/make_live_fixture.py <tujuan> [jumlah_produk]
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from defect_lab.export_onnx import export_backbone, export_head, quantize_weights  # noqa: E402
from defect_lab.patchcore import NearestHead, PatchEmbedder, to_tensor  # noqa: E402

out = Path(sys.argv[1]).resolve()
n_classes = int(sys.argv[2]) if len(sys.argv) > 2 else 2
dist = ROOT / "web" / "dist"
if out.exists():
    shutil.rmtree(out)
shutil.copytree(dist, out)
(out / "models").mkdir(exist_ok=True)

torch.manual_seed(0)
model = PatchEmbedder("resnet18", 384, weights=None)
summary = json.loads((out / "data" / "summary.json").read_text())
classes = summary["classes"][:n_classes]

fp32 = out / "models" / "_fp32.onnx"
export_backbone(model, fp32)
int8 = out / "models" / "backbone_int8.onnx"
quantize_weights(fp32, int8)
fp32.unlink()
sb = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])

for ci, cls in enumerate(classes):
    g = torch.Generator().manual_seed(ci)
    head = NearestHead(torch.randn(4096, 384, generator=g), half_bank=True)
    export_head(head, out / "models" / f"head_{cls}.onnx")
    hpath = str(out / "models" / f"head_{cls}.onnx")
    sh = ort.InferenceSession(hpath, providers=["CPUExecutionProvider"])
    path = out / "data" / "samples" / f"{cls}.json"
    data = json.loads(path.read_text())
    scores = []
    for s in data["items"]:
        x = np.asarray(Image.open(out / s["input"]).convert("RGB"))[None]
        emb = sb.run(None, {"image": to_tensor(x).numpy()})[0]
        sc = float(sh.run(None, {"patches": emb})[0].max())
        s["scores"]["resnet18"] = sc
        s["scores"]["resnet18_int8"] = sc
        s["maps"]["resnet18"] = s["maps"][summary["primary"]]
        s["iou"]["resnet18"] = None
        scores.append(sc)
    data["thresholds"]["resnet18"] = float(np.median(scores))
    path.write_text(json.dumps(data))

summary["live_method"] = "resnet18"
summary["classes"] = classes
(out / "data" / "summary.json").write_text(json.dumps(summary))
print(f"fixture bobot acak di {out} ({len(classes)} produk). JANGAN dipublikasikan.")
