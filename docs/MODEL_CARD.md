# Model card: Visual Defect Lab

## Ringkasan

Tiga detektor cacat visual tanpa label cacat (hanya belajar dari foto normal), satu model per produk VisA:

| Model | Fitur patch | Memori normal | Untuk |
|---|---|---|---|
| Klasik | OpenCV: mean dan std Lab, histogram orientasi Sobel 8 arah, LBP uniform 10 pola (24 angka) | coreset 2% (maks 20.000) | baseline |
| PatchCore ResNet18 | ImageNet torchvision, layer2 + layer3 beku, rata-rata 3 x 3, 384 angka | coreset 1% (maks 4.096), float16 | model browser (ONNX, bobot int8) |
| PatchCore WideResNet50 | ImageNet torchvision, layer2 + layer3 beku, dipadatkan ke 1.024 angka | coreset 1% (maks 20.000) | model paling akurat (precomputed) |

Tidak ada fine-tuning. "Fit" berarti mengumpulkan fitur patch normal dan memilih coreset k-center greedy.

## Penggunaan yang dimaksud

Demo portofolio dan bahan belajar tentang deteksi anomali visual: membandingkan fitur klasik dengan fitur transfer learning secara jujur, dengan lokalisasi cacat dan ketidakpastian.

## Bukan untuk

- Inspeksi kualitas produksi tanpa validasi ulang pada kamera, pencahayaan, dan produk sendiri.
- Produk di luar 12 produk VisA. Foto lain akan tampak "cacat" di mana-mana karena berbeda dari memori normal.
- Mengukur jenis cacat; model hanya menyatakan seberapa berbeda sebuah area dari normal.

## Hasil

Rata-rata 12 produk pada 2.162 foto uji (selang 95% bootstrap):

| Model | image AUROC | pixel AUROC | AUPRO | Cacat lolos | Normal ditolak | Waktu per foto (CPU) |
|---|---|---|---|---|---|---|
| Klasik | 0,811 (0,793 sampai 0,828) | 0,915 | 0,716 | 45,7% | 7,1% | 79 ms |
| PatchCore ResNet18 | 0,923 (0,912 sampai 0,934) | 0,980 | 0,858 | 22,4% | 7,6% | 63 ms |
| PatchCore WideResNet50 | 0,916 (0,905 sampai 0,927) | 0,981 | 0,865 | 23,3% | 7,9% | 283 ms |
| ResNet18 di browser (ONNX) | 0,922 | 0,979 | | | | sekitar 170 sampai 350 ms |

Per produk dan pembacaan: [HASIL_PER_PRODUK.md](HASIL_PER_PRODUK.md), [EVALUATION_REPORT.md](EVALUATION_REPORT.md).

## Ambang

Persentil ke-95 skor foto normal validasi per produk. Ambang ini menentukan keputusan tolak atau lolos dan area yang ditandai. Tingkat alarm palsu pada uji bisa berbeda dari 5%.

## Risiko dan batasan

- Cacat di bawah 0,2% luas foto sering terlewat pada resolusi 256 x 256.
- Produk dengan banyak benda dalam satu foto (kapsul, makaroni) atau latar bertekstur lebih sulit.
- Model browser memakai bobot int8 (perhitungan float32): image AUROC makro turun 0,0015 dan keputusan berubah pada 26 dari 2.162 foto uji (`evaluation/onnx_parity.json`). Kuantisasi statis bobot dan aktivasi tidak lolos batas kesetiaan pada data validasi.
- Kapsul dan makaroni 2 tetap lemah (image AUROC 0,62 sampai 0,77). Tingkat alarm palsu di uji 7 sampai 8%, sedikit di atas target 5% dari validasi.
- Bobot ImageNet membawa bias ImageNet; tidak relevan untuk keputusan tentang orang, tetapi memengaruhi ciri yang dianggap "mirip".

## Lisensi

Kode MIT. Data dan turunan foto CC BY 4.0 (VisA). Bobot ImageNet torchvision BSD-3-Clause. Rincian di [NOTICE.md](../NOTICE.md).
