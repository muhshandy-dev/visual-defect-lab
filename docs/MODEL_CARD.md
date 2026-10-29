# Model card: Visual Defect Lab

## Ringkasan

Tiga detektor cacat visual tanpa label cacat (hanya belajar dari foto normal), satu model per produk VisA:

| Model | Fitur patch | Memori normal | Untuk |
|---|---|---|---|
| Klasik | OpenCV: mean dan std Lab, histogram orientasi Sobel 8 arah, LBP uniform 10 pola (24 angka) | coreset 2% (maks 20.000) | baseline |
| PatchCore ResNet18 | ImageNet torchvision, layer2 + layer3 beku, rata-rata 3 x 3, 384 angka | coreset 1% (maks 4.096), float16 | model browser (ONNX int8) |
| PatchCore WideResNet50 | ImageNet torchvision, layer2 + layer3 beku, dipadatkan ke 1.024 angka | coreset 1% (maks 20.000) | model paling akurat (precomputed) |

Tidak ada fine-tuning. "Fit" berarti mengumpulkan fitur patch normal dan memilih coreset k-center greedy.

## Penggunaan yang dimaksud

Demo portofolio dan bahan belajar tentang deteksi anomali visual: membandingkan fitur klasik dengan fitur transfer learning secara jujur, dengan lokalisasi cacat dan ketidakpastian.

## Bukan untuk

- Inspeksi kualitas produksi tanpa validasi ulang pada kamera, pencahayaan, dan produk sendiri.
- Produk di luar 12 produk VisA. Foto lain akan tampak "cacat" di mana-mana karena berbeda dari memori normal.
- Mengukur jenis cacat; model hanya menyatakan seberapa berbeda sebuah area dari normal.

## Hasil

Lihat [EVALUATION_REPORT.md](EVALUATION_REPORT.md). Per 29 Oktober 2026 hanya baseline klasik yang sudah dievaluasi (image AUROC makro 0,811). PatchCore belum dijalankan.

## Ambang

Persentil ke-95 skor foto normal validasi per produk. Ambang ini menentukan keputusan tolak atau lolos dan area yang ditandai. Tingkat alarm palsu pada uji bisa berbeda dari 5%.

## Risiko dan batasan

- Cacat di bawah 0,2% luas foto sering terlewat pada resolusi 256 x 256.
- Produk dengan banyak benda dalam satu foto (kapsul, makaroni) atau latar bertekstur lebih sulit.
- Kuantisasi int8 model browser dapat menggeser skor sedikit; dampaknya diukur di `evaluation/onnx_parity.json` setelah ekspor.
- Bobot ImageNet membawa bias ImageNet; tidak relevan untuk keputusan tentang orang, tetapi memengaruhi ciri yang dianggap "mirip".

## Lisensi

Kode MIT. Data dan turunan foto CC BY 4.0 (VisA). Bobot ImageNet torchvision BSD-3-Clause. Rincian di [NOTICE.md](../NOTICE.md).
