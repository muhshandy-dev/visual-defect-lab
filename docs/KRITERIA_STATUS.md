# Status terhadap kriteria penerimaan

Kriteria ditulis 29 Oktober 2026 sebelum koding model (`kriteria-penerimaan.md` di folder proyek). Status per 29 Oktober 2026, sebelum PatchCore dijalankan.

| No | Kriteria | Status | Bukti |
|---|---|---|---|
| K1 | Data VisA asli, split resmi, checksum, lisensi, data card | Terpenuhi | `defect-lab data` memeriksa SHA-256 arsip; `evaluation/split.csv`; `docs/DATA_CARD.md`; `NOTICE.md`; footer website |
| K2 | Tanpa kebocoran; ambang dari validasi; konfigurasi dikunci | Terpenuhi untuk klasik | `config.py`; `tools/verify_published.py` memeriksa tidak ada tumpang tindih dan ambang = kuantil 0,95 skor validasi |
| K3 | Baseline klasik OpenCV di 12 produk | Terpenuhi | `src/defect_lab/classical.py`, `evaluation/raw/classical/*` |
| K4 | PatchCore WRN50 dan ResNet18 di 12 produk | **Belum**: kode dan tes ada, belum dijalankan | butuh bobot ImageNet (diblokir jaringan sesi) |
| K5 | image AUROC, pixel AUROC, AUPRO, mIoU | Terpenuhi untuk klasik | `src/defect_lab/metrics.py`, `tests/test_metrics.py` (kasus hitung tangan, pembanding sklearn), pemeriksaan ulang independen |
| K6 | Selang 95% dan perbandingan berpasangan | Sebagian: selang ada; perbandingan berpasangan menunggu K4 | `src/defect_lab/report.py`, `evaluation/results.json` |
| K7 | Analisis kesalahan | Terpenuhi untuk klasik | halaman Kesalahan, `by_size` dan `by_defect` di `results.json`, `docs/EVALUATION_REPORT.md` |
| K8 | ONNX int8 + paritas dengan PyTorch | **Belum** pada model asli | `export_onnx.py`; tes paritas dengan bobot acak lulus (`tests/test_patchcore.py`) |
| K9 | Inferensi browser nyata + paritas + latensi | Mekanisme terbukti, model asli **belum** | fixture bobot acak: selisih 1,1e-7 x ambang, sekitar 300 ms per foto (`tools/make_live_fixture.py`, e2e) |
| K10 | Website charcoal + amber, galeri zoom, before/after, heatmap, label precomputed/langsung, responsif, tema | Terpenuhi untuk data klasik | e2e 115 pemeriksaan (3 lebar layar, 2 tema, axe, CSP) |
| K11 | Reproduksi, notebook Kaggle, folder evaluation, tes, CI, dokumen Indonesia | Sebagian: semua ditulis; CI belum pernah berjalan di GitHub (repo belum ada); notebook belum dijalankan di Kaggle | `Makefile`, `docs/REPRODUCE.md`, `.github/workflows/ci.yml`, `.github/workflows/reproduce.yml`, `notebooks/` |
| K12 | Kejujuran status | Terpenuhi | README, Metode, dan laporan menyebut PatchCore, Kaggle, dan deploy belum dijalankan |

## Penyimpangan dari rencana Master

- Tanpa GPU Kaggle: fit dan evaluasi di CPU. PatchCore tidak melakukan backpropagation, jadi metodenya sama; hanya waktu yang lebih lama.
- Tanpa fine-tuning backbone (sesuai PatchCore). Bila diperlukan model yang dilatih ulang, itu tambahan terpisah.

## Yang tersisa dan siapa yang mengerjakan

1. Shandy: buat repo `muhshandy-dev/visual-defect-lab` dan pasang Claude GitHub App.
2. Salah satu dari: Shandy mengunggah dua berkas bobot torchvision ke thread, **atau** setelah repo ada, menjalankan workflow "Reproduksi penuh" dari tab Actions (runner GitHub bisa mengunduh bobot dan VisA sendiri).
3. Claude: menjalankan PatchCore dan ekspor (`tools/run_transfer.sh`), memperbarui laporan, push, PR draf, memantau CI.
4. Shandy: menghubungkan Cloudflare Pages (`docs/DEPLOY_CLOUDFLARE.md`) dan memeriksa situs live.
