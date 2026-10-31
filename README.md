# Visual Defect Lab

Deteksi dan lokalisasi cacat visual pada produk manufaktur. Model hanya belajar dari foto produk yang bagus, lalu memberi skor pada foto baru dan menunjuk area yang mencurigakan. Data: **VisA** (12 produk, 10.821 foto, masker cacat; CC BY 4.0).

> **Status (31 Oktober 2026): model dan evaluasi selesai, website belum di-deploy.** Ketiga metode sudah dievaluasi pada 2.162 foto uji resmi VisA. PatchCore dijalankan di runner GitHub Actions (bukan Kaggle). Model browser ONNX sudah diekspor dan diuji di Chromium. Situs live belum ada sampai Cloudflare Pages dihubungkan.

## Yang dibandingkan

| Model | Cara mengubah patch menjadi angka | Peran |
|---|---|---|
| Klasik (baseline) | OpenCV: warna Lab, gradien Sobel, Local Binary Pattern (24 angka per patch) | pembanding |
| PatchCore ResNet18 | fitur beku ImageNet (layer2 + layer3), 384 angka per patch | model browser (ONNX, bobot int8) |
| PatchCore WideResNet50 | fitur beku ImageNet, 1.024 angka per patch | backbone besar, skor precomputed |

Ketiganya memakai penilai yang sama: jarak tiap patch ke patch normal terdekat (kNN pada coreset), peta skor diperhalus Gaussian, skor foto = nilai tertinggi peta. Ambang operasi = persentil ke-95 skor foto normal validasi. Foto uji tidak dipakai untuk memilih apa pun.

## Hasil (2.162 foto uji, rata-rata 12 produk, selang 95% bootstrap)

| Ukuran | Klasik | PatchCore ResNet18 | PatchCore WideResNet50 |
|---|---|---|---|
| image AUROC | 0,811 (0,793 sampai 0,828) | 0,923 (0,912 sampai 0,934) | 0,916 (0,905 sampai 0,927) |
| pixel AUROC | 0,915 | 0,980 | 0,981 |
| AUPRO (FPR sampai 0,3) | 0,716 | 0,858 | 0,865 |
| mIoU pada ambang operasi | 0,125 | 0,119 | 0,118 |
| Foto normal ditolak | 7,1% | 7,6% | 7,9% |
| Foto cacat lolos | 45,7% (43,1 sampai 48,3%) | 22,4% (20,3 sampai 24,6%) | 23,3% (21,4 sampai 25,2%) |

PatchCore mengalahkan klasik dengan selisih yang jauh dari nol (image AUROC +0,11, cacat lolos -23 poin). WideResNet50 dan ResNet18 setara. Produk tersulit: kapsul dan makaroni 2. Cacat di bawah 0,2% luas foto masih 37 sampai 41% lolos. Model browser: image AUROC 0,922 setelah ekspor (PyTorch 0,923), paritas browser vs Python 1,9e-6 kali ambang, sekitar 0,2 sampai 0,35 detik per foto. Rincian per produk: [docs/HASIL_PER_PRODUK.md](docs/HASIL_PER_PRODUK.md).

## Struktur

```
src/defect_lab/   data, fitur klasik, PatchCore, metrik, laporan, ekspor ONNX dan web, CLI
tests/            tes unit Python (metrik dihitung tangan, kNN brute force, paritas ONNX)
tools/            verify_published.py (pemeriksaan independen), render_report.py (tabel hasil), run_transfer.sh, make_live_fixture.py (uji mekanisme browser, bobot acak)
evaluation/       split.csv, raw/ (skor dan histogram per produk), results.json, onnx_parity.json, environment.txt, e2e.txt
notebooks/        notebook reproduksi Kaggle (belum dijalankan di Kaggle)
web/              React + Vite + TypeScript, onnxruntime-web
docs/             DATA_CARD, MODEL_CARD, EVALUATION_REPORT, KRITERIA_STATUS, ARCHITECTURE, REPRODUCE, DEPLOY_CLOUDFLARE
```

Cara menjalankan ulang: [docs/REPRODUCE.md](docs/REPRODUCE.md), atau workflow **Reproduksi penuh** (label `reproduksi` pada PR atau tab Actions: unduh data dan bobot, jalankan semua, commit hasil ke branch baru) dan **Ekspor model browser** (label `ekspor-onnx`). Laporan: [docs/EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md), tabel per produk dengan CI 95% (dibuat otomatis): [docs/HASIL_PER_PRODUK.md](docs/HASIL_PER_PRODUK.md), [docs/MODEL_CARD.md](docs/MODEL_CARD.md), status kriteria: [docs/KRITERIA_STATUS.md](docs/KRITERIA_STATUS.md). Atribusi data: [NOTICE.md](NOTICE.md). Kode: MIT.
