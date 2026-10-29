"""Konfigurasi yang dikunci sebelum evaluasi test.

Semua angka di sini ditetapkan dari literatur (PatchCore, Roth dkk. 2022) dan batas CPU
sesi pengembangan, bukan dari hasil pada data test. Mengubah nilai di sini berarti
menjalankan ulang seluruh evaluasi dan mencatat alasannya di docs/EVALUATION_REPORT.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
EVAL_DIR = ROOT / "evaluation"
MODELS_DIR = ROOT / "models"
WEIGHTS_DIR = DATA_DIR / "weights"
WEB_PUBLIC = ROOT / "web" / "public"

VISA_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
VISA_SHA256 = "2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362"
VISA_BYTES = 1_929_840_640

CLASSES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
)

# Bobot ImageNet resmi torchvision (IMAGENET1K_V1). Nilai kedua = awalan SHA-256 yang
# tertulis di nama berkas resmi; skrip memeriksa awalan ini sebelum memuat bobot.
TORCHVISION_WEIGHTS = {
    "resnet18": ("resnet18-f37072fd.pth", "f37072fd"),
    "wide_resnet50_2": ("wide_resnet50_2-95faca4d.pth", "95faca4d"),
}

# "cpu" (default, dipakai untuk hasil di repo) atau "cuda" (misalnya GPU Kaggle).
DEVICE = os.environ.get("DEFECT_LAB_DEVICE", "cpu")

SEED = 2026
IMAGE_SIZE = 256
# Bagian citra normal train yang disisihkan sebagai validasi (untuk ambang operasi).
VAL_FRACTION = 0.10
# Ambang operasi: kuantil skor citra normal validasi. 0,95 berarti sekitar 5% citra
# normal validasi akan ditolak. Ambang yang sama dipakai untuk masker piksel.
THRESHOLD_QUANTILE = 0.95
# Sigma Gaussian untuk menghaluskan peta skor (piksel pada resolusi 256).
SMOOTH_SIGMA = 4.0
BOOTSTRAP_REPS = 1000
PIXEL_BINS = 2048
AUPRO_FPR_LIMIT = 0.3


@dataclass(frozen=True)
class MethodConfig:
    key: str
    label: str
    kind: str  # "classical" atau "patchcore"
    backbone: str | None = None
    embed_dim: int | None = None
    coreset_ratio: float = 0.01
    coreset_max: int = 20_000
    projection_dim: int = 128


METHODS = {
    "classical": MethodConfig(
        key="classical",
        label="Klasik OpenCV (LBP + gradien + warna, kNN)",
        kind="classical",
        coreset_ratio=0.02,
        coreset_max=20_000,
        projection_dim=24,
    ),
    "resnet18": MethodConfig(
        key="resnet18",
        label="PatchCore ResNet18 (model browser)",
        kind="patchcore",
        backbone="resnet18",
        embed_dim=384,
        coreset_ratio=0.01,
        coreset_max=4096,
    ),
    "wrn50": MethodConfig(
        key="wrn50",
        label="PatchCore WideResNet50",
        kind="patchcore",
        backbone="wide_resnet50_2",
        embed_dim=1024,
        coreset_ratio=0.01,
        coreset_max=20_000,
    ),
}
