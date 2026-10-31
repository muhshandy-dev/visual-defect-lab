# Status terhadap kriteria penerimaan

Kriteria ditulis 29 Oktober 2026 sebelum koding model (`kriteria-penerimaan.md` di folder proyek). Status per 31 Oktober 2026, setelah PatchCore dan model browser dijalankan.

| No | Kriteria | Status | Bukti |
|---|---|---|---|
| K1 | Data VisA asli, split resmi, checksum, lisensi, data card | Terpenuhi | `defect-lab data` memeriksa SHA-256 arsip; `evaluation/split.csv`; `docs/DATA_CARD.md`; `NOTICE.md`; footer website |
| K2 | Tanpa kebocoran; ambang dari validasi; konfigurasi dikunci | Terpenuhi | `config.py` dikunci sebelum evaluasi uji; `tools/verify_published.py` memeriksa tidak ada tumpang tindih dan ambang = kuantil 0,95 skor validasi untuk 3 metode x 12 produk |
| K3 | Baseline klasik OpenCV di 12 produk | Terpenuhi | `src/defect_lab/classical.py`, `evaluation/raw/classical/*`; diulang di runner GitHub dengan selisih skor maks 1,2e-6 |
| K4 | PatchCore WRN50 dan ResNet18 di 12 produk | Terpenuhi (di runner GitHub, CPU) | `evaluation/raw/resnet18/*`, `evaluation/raw/wrn50/*`, `evaluation/environment.txt` (run 41125770404) |
| K5 | image AUROC, pixel AUROC, AUPRO, mIoU | Terpenuhi | `src/defect_lab/metrics.py`, `tests/test_metrics.py`, pemeriksaan ulang independen. Versi ambang oracle tidak dibuat (kriteria menyebutnya opsional sebagai rujukan) |
| K6 | Selang 95% dan perbandingan berpasangan | Terpenuhi | `evaluation/results.json`, `docs/HASIL_PER_PRODUK.md`: selang untuk semua metrik termasuk cacat lolos dan normal ditolak, selisih berpasangan PatchCore vs klasik dan WRN50 vs ResNet18 |
| K7 | Analisis kesalahan | Terpenuhi | halaman Kesalahan (jumlah dan selang per produk), `by_size` dan `by_defect` di `results.json`, `docs/EVALUATION_REPORT.md` |
| K8 | ONNX int8 + paritas dengan PyTorch | Terpenuhi, dengan penyimpangan cara kuantisasi (lihat bawah) | `evaluation/onnx_parity.json`: backbone 2,7 MB, image AUROC 0,923 menjadi 0,922 pada seluruh foto uji, 26 keputusan berubah dari 2.162 |
| K9 | Inferensi browser nyata + paritas + latensi | Terpenuhi | `evaluation/e2e.txt`: 24 foto, selisih browser vs Python maks 1,9e-6 x ambang, 170 sampai 200 ms per foto di runner; unggah foto sendiri juga diuji |
| K10 | Website charcoal + amber, galeri zoom, before/after, heatmap, label precomputed/langsung, responsif, tema | Terpenuhi secara lokal dan di CI; situs live belum ada | e2e 165 pemeriksaan (3 lebar layar, 2 tema, axe, CSP, inferensi langsung) |
| K11 | Reproduksi, notebook Kaggle, folder evaluation, tes, CI, dokumen Indonesia | Sebagian: CI hijau di GitHub, dua workflow reproduksi berjalan; notebook Kaggle belum dijalankan di Kaggle | `Makefile`, `docs/REPRODUCE.md`, `.github/workflows/*.yml`, `notebooks/` |
| K12 | Kejujuran status | Terpenuhi | README, Metode, dan laporan menyebut Kaggle dan situs live belum ada, dan mencatat kegagalan run pertama |

## Penyimpangan dari rencana

- **Komputasi di runner GitHub Actions (CPU), bukan Kaggle GPU.** Kaggle dan unduhan bobot tidak terjangkau dari lingkungan pengembangan. PatchCore tidak melakukan backpropagation, jadi metodenya sama; hanya waktu yang lebih lama. Notebook Kaggle disediakan tetapi belum dijalankan di Kaggle.
- **Kuantisasi model browser: bobot int8, perhitungan float32.** Rencana awal adalah kuantisasi statis bobot dan aktivasi. Pada run pertama varian itu merusak skor; setelah dibandingkan dengan aturan yang hanya memakai data validasi, varian bobot int8 yang lolos (selisih 0,023 kali ambang vs 0,067 untuk statis). Ukuran hampir sama (2,7 MB vs 2,7 MB). Ini perlu persetujuan Shandy bila "ONNX quantized" harus berarti aktivasi int8.
- Tanpa fine-tuning backbone (sesuai PatchCore).

## Yang tersisa

1. Shandy: memeriksa dan menyetujui penyimpangan kuantisasi di atas.
2. Shandy: menghubungkan Cloudflare Pages (`docs/DEPLOY_CLOUDFLARE.md`) dan memeriksa situs live.
3. Opsional: menjalankan notebook di Kaggle untuk melengkapi K11.
