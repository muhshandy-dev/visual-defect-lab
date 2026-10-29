# Arsitektur

```
VisA tar ──► data.py ──► cache 256 px (npz) ──► evaluate.py ──► evaluation/raw/<metode>/<produk>.npz
                         split.csv                │                 models/<metode>/<produk>.pt
                                                  │
           classical.py (OpenCV, 24 angka/patch) ─┤
           patchcore.py (ResNet18/WRN50 beku) ────┘
                                                  ▼
                       report.py (bootstrap, berpasangan) ──► evaluation/results.json
                       export_onnx.py (int8 + paritas)    ──► web/public/models, evaluation/onnx_parity.json
                       export_web.py                      ──► web/public/{data,img,in,mask,map}
                       tools/verify_published.py (independen, tanpa kode paket)
```

## Python (`src/defect_lab`)

- `config.py`: semua pengaturan yang dikunci sebelum evaluasi test.
- `data.py`: unduh dan verifikasi VisA, indeks split, cache.
- `classical.py`: deskriptor patch klasik (warna Lab, gradien Sobel, LBP).
- `patchcore.py`: `PatchEmbedder` (backbone beku, layer2 + layer3, rata-rata 3 x 3), `greedy_coreset` (k-center pada proyeksi acak), `NearestHead` (jarak tetangga terdekat, bilinear, Gaussian sigma 4). `NearestHead` dipakai untuk ketiga metode dan diekspor apa adanya ke ONNX, sehingga browser menjalankan rumus yang sama.
- `metrics.py`: AUROC eksak, AUROC dan AUPRO dari histogram, IoU.
- `report.py`: rata-rata makro, bootstrap bertingkat per produk, selisih berpasangan, analisis ukuran dan jenis cacat.

## Website (`web/`)

React 19 + Vite + TypeScript, tanpa pustaka UI. Rute berbasis hash (`#/inspeksi?produk=pcb1&foto=a012`).

- `src/lib/heat.ts`: logika murni peta skor (kodekan rasio skor terhadap ambang ke 8 bit, warna amber, tepi masker, IoU). Diuji di `tests/heat.test.ts` dan cocok dengan `export_web.encode`.
- `src/lib/infer.ts`: onnxruntime-web (WASM, satu thread). Backbone int8 dan kepala per produk dimuat saat dibutuhkan.
- `src/components/Viewer.tsx`: foto, lapisan peta skor, garis area yang ditandai dan cacat sebenarnya, pembanding sebelum dan sesudah, zoom roda/tombol/cubit, geser.
- Halaman: Inspeksi, Galeri, Kinerja, Kesalahan, Metode.

## Kontrak data website

- Peta skor: PNG abu-abu 256 x 256, `v = floor(min(skor/ambang, 2) / 2 x 255 + 0,5)`. `v >= 128` berarti di atas ambang.
- `data/summary.json`: angka dari `results.json` dan ringkasan ONNX.
- `data/samples/<produk>.json`: sampel (acak berbenih + kesalahan terbesar model utama), skor per metode, IoU, jalur berkas.
