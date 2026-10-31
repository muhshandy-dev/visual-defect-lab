# Laporan evaluasi

> **Status 31 Oktober 2026:** ketiga metode sudah dievaluasi pada data uji resmi VisA. Baseline klasik dijalankan di lingkungan pengembangan dan diulang di runner GitHub (skor sama, selisih maksimum 1,2e-6). PatchCore ResNet18 dan WideResNet50 dijalankan di runner GitHub (workflow "Reproduksi penuh", run 41125770404) karena bobot ImageNet tidak bisa diunduh dari lingkungan pengembangan. Model browser diekspor oleh workflow "Ekspor model browser" (run 41147062538). Tidak ada angka yang diperkirakan atau diambil dari makalah.

Tabel lengkap per produk dengan selang 95%, selisih berpasangan, dan paritas ONNX ada di [HASIL_PER_PRODUK.md](HASIL_PER_PRODUK.md). Berkas itu dibuat otomatis dari `evaluation/results.json` dan CI memeriksa keduanya tetap sama.

## Protokol

- Data: VisA, split resmi `1cls`, 12 produk. Fit 7.795 foto normal, validasi 864 foto normal (10% dari latih, acak berbenih), uji 2.162 foto (1.200 cacat dengan masker, 962 normal). Lihat [DATA_CARD.md](DATA_CARD.md).
- Resolusi 256 x 256 tanpa crop. Satu model per produk.
- Pengaturan dikunci di `src/defect_lab/config.py` sebelum evaluasi uji: coreset 2% (klasik, maks 20.000) dan 1% (PatchCore, maks 4.096 untuk ResNet18, 20.000 untuk WideResNet50), Gaussian sigma 4, ambang operasi = persentil ke-95 skor foto normal validasi. Tidak ada penyetelan pada data uji.
- Skor foto = nilai maksimum peta skor yang sudah dihaluskan, sehingga keputusan foto dan area yang ditandai memakai ambang yang sama.
- Metrik: image AUROC, pixel AUROC, AUPRO sampai FPR 0,3, mIoU (rata-rata IoU per foto cacat pada ambang operasi), tingkat tolak foto normal, tingkat lolos foto cacat. Rata-rata makro atas 12 produk.
- Ketidakpastian: 1.000 bootstrap bertingkat (normal dan cacat diambil ulang terpisah) per produk. Semua metode memakai indeks ulangan yang sama sehingga selisihnya berpasangan. Laju cacat lolos dan normal ditolak ikut dibootstrap, jadi keduanya juga punya selang 95% per produk dan makro.
- Pixel AUROC dan AUPRO dihitung dari histogram 2.048 bin kuantil per produk. Selisih terhadap perhitungan eksak dicatat di `evaluation/raw/<metode>/<produk>.json`.
- Semua angka diperiksa ulang oleh `tools/verify_published.py` tanpa memakai kode metrik paket (lulus untuk 3 metode x 12 produk).

## Hasil makro (2.162 foto uji)

| Metode | image AUROC | pixel AUROC | AUPRO | mIoU | Normal ditolak | Cacat lolos |
|---|---|---|---|---|---|---|
| Klasik OpenCV | 0,811 (0,793 sampai 0,828) | 0,915 (0,907 sampai 0,924) | 0,716 (0,698 sampai 0,737) | 0,125 (0,115 sampai 0,134) | 7,1% (5,5 sampai 8,9%) | 45,7% (43,1 sampai 48,3%) |
| PatchCore ResNet18 | 0,923 (0,912 sampai 0,934) | 0,980 (0,977 sampai 0,982) | 0,858 (0,845 sampai 0,872) | 0,119 (0,110 sampai 0,127) | 7,6% (5,9 sampai 9,3%) | 22,4% (20,3 sampai 24,6%) |
| PatchCore WideResNet50 | 0,916 (0,905 sampai 0,927) | 0,981 (0,979 sampai 0,983) | 0,865 (0,851 sampai 0,878) | 0,118 (0,111 sampai 0,126) | 7,9% (6,1 sampai 9,5%) | 23,3% (21,4 sampai 25,2%) |

Selisih berpasangan (rata-rata makro, selang 95% bootstrap):

- ResNet18 vs klasik: image AUROC +0,112 (0,094 sampai 0,130), AUPRO +0,142 (0,123 sampai 0,161), cacat lolos -23,3 poin (-26,2 sampai -20,6). mIoU -0,006 (-0,016 sampai 0,005), tidak berbeda.
- WideResNet50 vs ResNet18: image AUROC -0,007 (-0,015 sampai 0,003), AUPRO +0,006 (-0,002 sampai 0,015). Tidak ada perbedaan yang meyakinkan.

## Menurut ukuran cacat (bagian tertangkap pada ambang operasi)

| Ukuran cacat | Foto | Klasik | ResNet18 | WideResNet50 |
|---|---|---|---|---|
| kecil (< 0,2% luas foto) | 619 | 36,8% | 63,3% | 58,8% |
| sedang (0,2 sampai 1%) | 388 | 67,8% | 89,7% | 94,1% |
| besar (> 1%) | 193 | 83,4% | 99,0% | 99,0% |

## Pembacaan

- **Transfer learning jelas lebih baik daripada fitur tangan.** Kedua PatchCore menaikkan image AUROC sekitar 0,11 dan AUPRO sekitar 0,14, dan memotong cacat yang lolos dari 45,7% menjadi sekitar 22 sampai 23% pada tingkat alarm palsu yang setara (sekitar 7 sampai 8%). Semua selang selisihnya jauh dari nol.
- **WideResNet50 tidak mengalahkan ResNet18 di sini.** Rata-ratanya setara, dan per produk saling menang: WideResNet50 lebih baik pada mete, PCB 1 dan PCB 3, tetapi jauh lebih buruk pada makaroni 2 (0,622 vs 0,772). Penjelasan yang mungkin: pada 256 px, fitur 1.024 dimensi dipadatkan dengan rata-rata saluran dan tekstur latar makaroni 2 mendominasi. Ini dugaan, belum diuji. Karena itu ResNet18 dipakai sebagai model browser tanpa mengorbankan akurasi rata-rata, dan WideResNet50 hampir 4,5 kali lebih lambat per foto di CPU (283 vs 63 ms).
- **Produk sulit tetap sulit.** Kapsul (0,731 dan 0,748) dan makaroni 2 (0,772 dan 0,622) tertinggal: banyak benda kecil dalam satu foto dan latar bertekstur. Di sisi lain, permen karet, fryum pipa, dan PCB 4 di atas 0,98 pada kedua backbone.
- **Cacat kecil adalah kelemahan utama.** Separuh foto cacat punya area di bawah 0,2% luas foto, dan 37 sampai 41% di antaranya tetap lolos pada PatchCore. Pada 256 x 256, satu patch fitur layer3 mencakup sekitar 16 x 16 piksel, lebih besar dari banyak cacat ini.
- **Lokalisasi tajam per peringkat, tetapi area pada ambang operasi kurang pas.** Pixel AUROC naik ke 0,98 dan AUPRO ke 0,86, tetapi mIoU tidak naik (0,12). Ambang piksel sengaja sama dengan ambang foto (persentil 95 normal validasi), dan peta yang dihaluskan Gaussian cenderung menandai area lebih lebar dari cacat kecil. Ambang khusus piksel yang dipilih dari validasi akan menjadi perbaikan berikutnya; tidak dilakukan di sini agar tidak menyetel setelah melihat hasil uji.
- **Alarm palsu di uji sedikit di atas target validasi.** Target 5% dari validasi menjadi 7 sampai 8% di uji, bervariasi per produk karena tiap produk hanya punya 45 sampai 90 foto validasi. Selang per produk ada di HASIL_PER_PRODUK.md.

## Model browser (ONNX)

- Model browser = PatchCore ResNet18: backbone ONNX 2,7 MB (float32 10,6 MB) + memori patch per produk 3,0 MB (float16), diunduh hanya untuk produk yang dihitung.
- Dua cara kuantisasi dibandingkan. Pemilihan hanya memakai foto normal validasi: varian terkecil dengan selisih skor maksimum di bawah 0,05 kali ambang. Bobot int8 saja (perhitungan float32): 0,023, lolos. Kuantisasi statis bobot dan aktivasi int8: 0,067, tidak lolos. Yang dipakai: bobot int8.
- Pada seluruh 2.162 foto uji: image AUROC makro 0,923 (PyTorch) menjadi 0,922 (ONNX), pixel AUROC 0,980 menjadi 0,979, keputusan berubah pada 26 foto (1,2%).
- Paritas browser: onnxruntime-web di Chromium vs onnxruntime Python pada 24 foto (2 per produk), selisih maksimum 1,9e-6 kali ambang. Latensi di runner GitHub sekitar 170 sampai 200 ms per foto, ditambah sekitar 0,3 detik memuat model pertama kali.
- Catatan jujur: run reproduksi pertama memakai kuantisasi statis dan hasilnya buruk (contoh lilin image AUROC 0,98 menjadi 0,56; uji e2e gagal 28 pemeriksaan tetapi job tetap hijau karena keluaran e2e dialirkan ke `tee` tanpa `pipefail`). Temuan ini (termasuk angka uji) yang memicu perbandingan dua varian; aturan pemilihannya sendiri hanya memakai data validasi. Pada run ekspor berikutnya varian statis jauh lebih baik (AUROC uji 0,92) tetapi tetap tidak lolos batas kesetiaan validasi. Penyebab perbedaan antar run untuk varian statis belum dipastikan. `pipefail` sudah ditambahkan.
- Pemeriksaan determinisme: workflow ekspor memfit ulang ResNet18 dan membandingkan skornya dengan hasil yang di-commit; selisih maksimum 0 (`evaluation/onnx_refit_check.txt`).

## Lingkungan

`evaluation/environment.txt`: Python 3.13, torch 2.15 CPU, torchvision 0.29.2, timm 1.0.30, onnxruntime 1.31.1, OpenCV 5.0, runner GitHub 4 vCPU. Bobot ImageNet torchvision dengan SHA-256 tercatat. Waktu fit seluruh 12 produk: klasik 20 menit, ResNet18 31 menit, WideResNet50 69 menit.
