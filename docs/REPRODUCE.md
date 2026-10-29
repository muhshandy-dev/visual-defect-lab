# Reproduksi

Semua angka di README, `evaluation/`, dan website dapat dibuat ulang dengan perintah di bawah. Butuh sekitar 5 GB ruang disk dan 16 GB RAM. Di CPU 4 inti, seluruh rangkaian kira-kira satu jam.

```bash
make install                      # .venv + paket + PyTorch + timm + onnxruntime
mkdir -p data/weights             # bobot ImageNet resmi torchvision
curl -L -o data/weights/resnet18-f37072fd.pth https://download.pytorch.org/models/resnet18-f37072fd.pth
curl -L -o data/weights/wide_resnet50_2-95faca4d.pth https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth
make data                         # unduh VisA 1,9 GB, cek SHA-256, cache 256 px, tulis evaluation/split.csv
make evaluate                     # klasik, ResNet18, WideResNet50 untuk 12 produk
make report                       # evaluation/results.json (bootstrap 1000 kali)
make onnx                         # web/public/models + evaluation/onnx_parity.json
make web                          # web/public/data, foto, masker, peta skor
make verify                       # pemeriksaan independen
cd web && npm ci && npm run build && npm run e2e
```

Bobot diperiksa dengan awalan SHA-256 yang tertulis di nama berkas resmi torchvision. GPU opsional: `DEFECT_LAB_DEVICE=cuda`. Hasil di repositori dibuat dengan `cpu`; di GPU angka bisa berbeda sangat sedikit karena urutan operasi floating point.

## Kaggle

`notebooks/visual-defect-lab-kaggle.ipynb` menjalankan langkah yang sama di Kaggle dan membandingkan hasilnya dengan `evaluation/results.json`. Notebook ini belum pernah dijalankan di Kaggle.

## Determinisme

- Pembagian val, pemilihan coreset, proyeksi acak, bootstrap, dan pemilihan sampel website memakai benih tetap (`SEED = 2026` di `src/defect_lab/config.py`).
- `defect-lab export-web` menghitung ulang peta skor sampel dari memori patch yang tersimpan dan gagal bila skor berbeda dari hasil evaluasi lebih dari 0,001 kali ambang.
