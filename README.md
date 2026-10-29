# Visual Defect Lab

Deteksi dan lokalisasi cacat visual pada produk manufaktur. Model hanya belajar dari foto produk yang bagus, lalu memberi skor pada foto baru dan menunjuk area yang mencurigakan. Data: **VisA** (12 produk, 10.821 foto, masker cacat; CC BY 4.0).

> **Status (29 Oktober 2026): dalam pengerjaan.** Baseline klasik sudah dievaluasi pada data asli. PatchCore (transfer learning) dan model ONNX browser **belum dijalankan** karena bobot ImageNet belum bisa diunduh dari lingkungan pengembangan. Angka di bawah hanya untuk baseline klasik. Website belum di-deploy.

## Yang dibandingkan

| Model | Cara mengubah patch menjadi angka | Status |
|---|---|---|
| Klasik (baseline) | OpenCV: warna Lab, gradien Sobel, Local Binary Pattern (24 angka per patch) | selesai |
| PatchCore ResNet18 | fitur beku ImageNet (layer2 + layer3), model browser ONNX int8 | menunggu bobot |
| PatchCore WideResNet50 | fitur beku ImageNet, model paling akurat | menunggu bobot |

Ketiganya memakai penilai yang sama: jarak tiap patch ke patch normal terdekat (kNN pada coreset), peta skor diperhalus Gaussian, skor foto = nilai tertinggi peta. Ambang operasi = persentil ke-95 skor foto normal validasi. Foto uji tidak dipakai untuk memilih apa pun.

## Hasil baseline klasik (2.162 foto uji, rata-rata 12 produk)

| Ukuran | Nilai | Selang 95% |
|---|---|---|
| image AUROC | 0,811 | 0,793 sampai 0,828 |
| pixel AUROC | 0,915 | 0,907 sampai 0,924 |
| AUPRO (FPR sampai 0,3) | 0,716 | 0,698 sampai 0,737 |
| mIoU pada ambang operasi | 0,125 | 0,115 sampai 0,134 |
| Foto normal ditolak | 7,1% | |
| Foto cacat lolos | 45,7% | |

Selang dari 1.000 bootstrap bertingkat per produk. Produk terlemah: makaroni 2 (image AUROC 0,526), terkuat: PCB 1 (0,930). Cacat kecil (di bawah 0,2% luas foto, 619 dari 1.200 foto cacat) hanya 37% tertangkap.

## Struktur

```
src/defect_lab/   data, fitur klasik, PatchCore, metrik, laporan, ekspor ONNX dan web, CLI
tests/            tes unit Python (metrik dihitung tangan, kNN brute force, paritas ONNX)
tools/            verify_published.py (pemeriksaan independen), run_transfer.sh, make_live_fixture.py (uji mekanisme browser, bobot acak)
evaluation/       split.csv, raw/ (skor dan histogram per produk), results.json
notebooks/        notebook reproduksi Kaggle (belum dijalankan di Kaggle)
web/              React + Vite + TypeScript, onnxruntime-web
docs/             DATA_CARD, MODEL_CARD, EVALUATION_REPORT, KRITERIA_STATUS, ARCHITECTURE, REPRODUCE, DEPLOY_CLOUDFLARE
```

Cara menjalankan ulang: [docs/REPRODUCE.md](docs/REPRODUCE.md), atau workflow **Reproduksi penuh** di tab Actions (unduh data dan bobot, jalankan semua, commit hasil ke branch baru). Laporan: [docs/EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md), [docs/MODEL_CARD.md](docs/MODEL_CARD.md), status kriteria: [docs/KRITERIA_STATUS.md](docs/KRITERIA_STATUS.md). Atribusi data: [NOTICE.md](NOTICE.md). Kode: MIT.
