#!/usr/bin/env bash
# Jalankan sisa pipeline setelah bobot ImageNet tersedia (di sesi pengembangan atau lokal).
# Pemakaian: tools/run_transfer.sh <resnet18-f37072fd.pth> <wide_resnet50_2-95faca4d.pth>
# Data VisA dan cache harus sudah ada (defect-lab data). Baseline klasik tidak dijalankan ulang
# bila hasilnya sudah ada di evaluation/raw/classical.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
mkdir -p data/weights
cp "$1" data/weights/resnet18-f37072fd.pth
cp "$2" data/weights/wide_resnet50_2-95faca4d.pth
sha256sum data/weights/*.pth

[ -f evaluation/raw/classical/pipe_fryum.npz ] || python -m defect_lab.cli evaluate --method classical
python -m defect_lab.cli evaluate --method resnet18
python -m defect_lab.cli evaluate --method wrn50
python -m defect_lab.cli report
python -m defect_lab.cli export-onnx
python -m defect_lab.cli export-web
python tools/verify_published.py
(cd web && npm run build && npm run e2e)
