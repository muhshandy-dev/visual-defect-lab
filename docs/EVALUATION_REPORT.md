# Laporan evaluasi

> **Status 29 Oktober 2026:** hanya baseline klasik yang sudah dievaluasi. PatchCore ResNet18, PatchCore WideResNet50, dan model ONNX int8 **belum dijalankan** karena bobot ImageNet belum dapat diunduh dari lingkungan pengembangan. Bagian yang belum ada hasilnya ditandai "belum dijalankan". Tidak ada angka yang diperkirakan atau diambil dari makalah.

## Protokol

- Data: VisA, split resmi `1cls`, 12 produk. Fit 7.795 foto normal, validasi 864 foto normal (10% dari latih, acak berbenih), uji 2.162 foto (1.200 cacat dengan masker, 962 normal). Lihat [DATA_CARD.md](DATA_CARD.md).
- Resolusi 256 x 256 tanpa crop. Satu model per produk.
- Pengaturan dikunci di `src/defect_lab/config.py` sebelum evaluasi uji: coreset 2% (klasik, maks 20.000) dan 1% (PatchCore, maks 4.096 untuk ResNet18, 20.000 untuk WideResNet50), Gaussian sigma 4, ambang operasi = persentil ke-95 skor foto normal validasi. Tidak ada penyetelan pada data uji.
- Skor foto = nilai maksimum peta skor yang sudah dihaluskan, sehingga keputusan foto dan area yang ditandai memakai ambang yang sama.
- Metrik: image AUROC, pixel AUROC, AUPRO sampai FPR 0,3, mIoU (rata-rata IoU per foto cacat pada ambang operasi), tingkat tolak foto normal, tingkat lolos foto cacat. Rata-rata makro atas 12 produk.
- Ketidakpastian: 1.000 bootstrap bertingkat (normal dan cacat diambil ulang terpisah) per produk. Semua metode memakai indeks ulangan yang sama sehingga selisihnya berpasangan.
- Pixel AUROC dan AUPRO dihitung dari histogram 2.048 bin kuantil per produk. Selisih pixel AUROC histogram terhadap perhitungan eksak pada semua piksel dicatat di `evaluation/raw/<metode>/<produk>.json` (untuk klasik, paling besar 2,6e-5).
- Semua angka diperiksa ulang oleh `tools/verify_published.py` tanpa memakai kode metrik paket.

## Hasil makro

| Metode | image AUROC | pixel AUROC | AUPRO | mIoU | Normal ditolak | Cacat lolos |
|---|---|---|---|---|---|---|
| Klasik OpenCV | 0,811 (0,793 sampai 0,828) | 0,915 (0,907 sampai 0,924) | 0,716 (0,698 sampai 0,737) | 0,125 (0,115 sampai 0,134) | 7,1% | 45,7% |
| PatchCore ResNet18 | belum dijalankan | | | | | |
| PatchCore WideResNet50 | belum dijalankan | | | | | |

## Baseline klasik per produk

| Produk | image AUROC (selang 95%) | pixel AUROC | AUPRO | mIoU | Normal ditolak | Cacat lolos |
|---|---|---|---|---|---|---|
| Lilin | 0,805 (0,740 sampai 0,863) | 0,876 | 0,750 | 0,112 | 8% | 44% |
| Kapsul | 0,711 (0,629 sampai 0,786) | 0,966 | 0,719 | 0,106 | 12% | 57% |
| Kacang mete | 0,862 (0,804 sampai 0,916) | 0,754 | 0,669 | 0,123 | 2% | 48% |
| Permen karet | 0,859 (0,798 sampai 0,913) | 0,952 | 0,536 | 0,300 | 6% | 33% |
| Kerupuk fryum | 0,856 (0,794 sampai 0,908) | 0,882 | 0,742 | 0,094 | 10% | 30% |
| Makaroni 1 | 0,715 (0,642 sampai 0,783) | 0,929 | 0,689 | 0,061 | 5% | 73% |
| Makaroni 2 | 0,526 (0,443 sampai 0,605) | 0,889 | 0,577 | 0,002 | 12% | 89% |
| PCB 1 | 0,930 (0,887 sampai 0,963) | 0,984 | 0,827 | 0,224 | 7% | 21% |
| PCB 2 | 0,922 (0,883 sampai 0,957) | 0,947 | 0,834 | 0,149 | 1% | 32% |
| PCB 3 | 0,837 (0,781 sampai 0,885) | 0,944 | 0,785 | 0,143 | 13% | 41% |
| PCB 4 | 0,866 (0,815 sampai 0,911) | 0,894 | 0,588 | 0,052 | 6% | 46% |
| Kerupuk pipa | 0,846 (0,778 sampai 0,904) | 0,964 | 0,878 | 0,131 | 4% | 34% |

## Menurut ukuran cacat (klasik)

| Ukuran cacat | Foto | Tertangkap | mIoU |
|---|---|---|---|
| kecil (< 0,2% luas foto) | 619 | 37% | 0,054 |
| sedang (0,2 sampai 1%) | 388 | 68% | 0,175 |
| besar (> 1%) | 193 | 83% | 0,250 |

## Pembacaan baseline klasik

- Baseline klasik cukup memisahkan foto pada produk yang posisinya tetap (PCB 1 dan 2, image AUROC di atas 0,92), tetapi gagal pada makaroni 2 (0,526, selang 95% mencakup 0,5): latar hijau bertekstur membuat patch normal sangat beragam, sehingga noda kecil tidak menonjol.
- Pixel AUROC tinggi (0,915) tetapi mIoU rendah (0,125). Sebagian besar piksel normal memang mudah, sedangkan area yang ditandai pada ambang operasi jarang menempel tepat pada cacat. Inilah alasan AUPRO dan mIoU dilaporkan, bukan hanya pixel AUROC.
- Pada ambang operasi, 45,7% foto cacat lolos. Cacat kecil (separuh dari semua foto cacat) hanya 37% tertangkap. Untuk inspeksi nyata baseline ini tidak layak dipakai; ia berfungsi sebagai pembanding.
- Tingkat tolak foto normal pada uji (7,1%) sedikit di atas target 5% dari validasi, dan bervariasi 1% sampai 13% per produk karena tiap produk hanya punya 45 sampai 90 foto validasi.

## Yang akan ditambahkan setelah PatchCore dijalankan

- Baris PatchCore pada tabel makro dan per produk, selisih berpasangan terhadap klasik dan WideResNet50 terhadap ResNet18.
- Paritas ONNX int8: ukuran berkas, perubahan image/pixel AUROC dibanding PyTorch pada seluruh foto uji, jumlah keputusan yang berubah (`evaluation/onnx_parity.json`).
- Paritas browser (onnxruntime-web vs onnxruntime Python) dan latensi dari tes e2e.

## Uji mekanisme inferensi browser (bukan hasil model)

Sebelum bobot asli tersedia, jalur browser diuji dengan fixture berbobot acak (`tools/make_live_fixture.py`, tidak dipublikasikan): backbone ResNet18 acak dikuantisasi int8 dengan cara yang sama, kepala dengan bank acak, dua produk, empat foto. Di Chromium headless, onnxruntime-web 1.30 berjalan di bawah CSP situs, selisih skor browser terhadap onnxruntime Python maksimum 1,1e-7 kali ambang, dan satu inferensi sekitar 300 ms (pertama kali ditambah sekitar 0,5 sampai 0,7 detik memuat model). Angka skor dari fixture ini tidak bermakna; yang diuji hanya mekanismenya.
