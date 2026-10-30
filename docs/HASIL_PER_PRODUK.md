# Hasil per produk

Berkas ini dibuat otomatis oleh `tools/render_report.py` dari `evaluation/results.json`. Jangan diedit tangan.

Semua angka dihitung pada data uji resmi VisA (split 1cls). CI 95% berasal dari bootstrap bertingkat 1000 kali (foto normal dan foto cacat diambil ulang terpisah di tiap produk). Ambang operasi tiap produk ditetapkan dari kuantil 0,95 skor foto normal validasi, tidak pernah dari data uji, jadi laju cacat lolos dan normal ditolak adalah angka jujur untuk ambang yang benar-benar akan dipakai.

## Rata-rata makro 12 produk

| Metrik | Klasik OpenCV (LBP + gradien + warna, kNN) | PatchCore ResNet18 (model browser) | PatchCore WideResNet50 |
|---|---|---|---|
| Image AUROC | 0,811 (0,793 sampai 0,828) | 0,923 (0,912 sampai 0,934) | 0,916 (0,905 sampai 0,927) |
| Pixel AUROC | 0,915 (0,907 sampai 0,924) | 0,980 (0,977 sampai 0,982) | 0,981 (0,979 sampai 0,983) |
| AUPRO (FPR 0,3) | 0,716 (0,698 sampai 0,737) | 0,858 (0,845 sampai 0,872) | 0,865 (0,851 sampai 0,878) |
| mIoU | 0,125 (0,115 sampai 0,134) | 0,119 (0,110 sampai 0,127) | 0,118 (0,111 sampai 0,126) |
| Normal ditolak | 7,1% (5,5% sampai 8,9%) | 7,6% (5,9% sampai 9,3%) | 7,9% (6,1% sampai 9,5%) |
| Cacat lolos | 45,7% (43,1% sampai 48,3%) | 22,4% (20,3% sampai 24,6%) | 23,3% (21,4% sampai 25,2%) |

## Selisih berpasangan

Selisih rata-rata makro dengan sampel bootstrap yang sama untuk kedua metode. Untuk AUROC, AUPRO dan mIoU, positif berarti lebih baik; untuk cacat lolos dan normal ditolak, negatif berarti lebih baik.

| Perbandingan | Metrik | Selisih | CI 95% |
|---|---|---|---|
| PatchCore ResNet18 (model browser) vs Klasik OpenCV (LBP + gradien + warna, kNN) | Image AUROC | 0,112 | 0,094 sampai 0,130 |
| PatchCore ResNet18 (model browser) vs Klasik OpenCV (LBP + gradien + warna, kNN) | Pixel AUROC | 0,065 | 0,056 sampai 0,073 |
| PatchCore ResNet18 (model browser) vs Klasik OpenCV (LBP + gradien + warna, kNN) | AUPRO (FPR 0,3) | 0,142 | 0,123 sampai 0,161 |
| PatchCore ResNet18 (model browser) vs Klasik OpenCV (LBP + gradien + warna, kNN) | mIoU | -0,006 | -0,016 sampai 0,005 |
| PatchCore ResNet18 (model browser) vs Klasik OpenCV (LBP + gradien + warna, kNN) | Normal ditolak | 0,5% | -1,6% sampai 2,7% |
| PatchCore ResNet18 (model browser) vs Klasik OpenCV (LBP + gradien + warna, kNN) | Cacat lolos | -23,3% | -26,2% sampai -20,6% |
| PatchCore WideResNet50 vs Klasik OpenCV (LBP + gradien + warna, kNN) | Image AUROC | 0,105 | 0,088 sampai 0,124 |
| PatchCore WideResNet50 vs Klasik OpenCV (LBP + gradien + warna, kNN) | Pixel AUROC | 0,066 | 0,057 sampai 0,074 |
| PatchCore WideResNet50 vs Klasik OpenCV (LBP + gradien + warna, kNN) | AUPRO (FPR 0,3) | 0,149 | 0,128 sampai 0,167 |
| PatchCore WideResNet50 vs Klasik OpenCV (LBP + gradien + warna, kNN) | mIoU | -0,006 | -0,016 sampai 0,004 |
| PatchCore WideResNet50 vs Klasik OpenCV (LBP + gradien + warna, kNN) | Normal ditolak | 0,7% | -1,6% sampai 3,0% |
| PatchCore WideResNet50 vs Klasik OpenCV (LBP + gradien + warna, kNN) | Cacat lolos | -22,3% | -25,2% sampai -19,7% |
| PatchCore WideResNet50 vs PatchCore ResNet18 (model browser) | Image AUROC | -0,007 | -0,015 sampai 0,003 |
| PatchCore WideResNet50 vs PatchCore ResNet18 (model browser) | Pixel AUROC | 0,001 | -0,001 sampai 0,003 |
| PatchCore WideResNet50 vs PatchCore ResNet18 (model browser) | AUPRO (FPR 0,3) | 0,006 | -0,002 sampai 0,015 |
| PatchCore WideResNet50 vs PatchCore ResNet18 (model browser) | mIoU | -0,001 | -0,006 sampai 0,004 |
| PatchCore WideResNet50 vs PatchCore ResNet18 (model browser) | Normal ditolak | 0,2% | -1,4% sampai 1,8% |
| PatchCore WideResNet50 vs PatchCore ResNet18 (model browser) | Cacat lolos | 0,9% | -1,1% sampai 2,7% |

## Klasik OpenCV (LBP + gradien + warna, kNN)

| Produk | Image AUROC (CI 95%) | Pixel AUROC | AUPRO | Cacat lolos (CI 95%) | Normal ditolak (CI 95%) |
|---|---|---|---|---|---|
| Makaroni 2 | 0,526 (0,443 sampai 0,605) | 0,889 | 0,577 | 89/100 = 89,0% (83,0% sampai 95,0%) | 12/100 = 12,0% (6,0% sampai 19,0%) |
| Kapsul | 0,711 (0,629 sampai 0,786) | 0,966 | 0,719 | 57/100 = 57,0% (48,0% sampai 67,0%) | 7/60 = 11,7% (3,3% sampai 20,0%) |
| Makaroni 1 | 0,715 (0,642 sampai 0,783) | 0,929 | 0,689 | 73/100 = 73,0% (64,0% sampai 82,0%) | 5/100 = 5,0% (1,0% sampai 10,0%) |
| Lilin | 0,805 (0,740 sampai 0,863) | 0,876 | 0,750 | 44/100 = 44,0% (35,0% sampai 55,0%) | 8/100 = 8,0% (3,0% sampai 14,0%) |
| PCB 3 | 0,837 (0,781 sampai 0,885) | 0,944 | 0,785 | 41/100 = 41,0% (33,0% sampai 51,0%) | 13/101 = 12,9% (6,9% sampai 19,8%) |
| Fryum pipa | 0,846 (0,778 sampai 0,904) | 0,964 | 0,878 | 34/100 = 34,0% (25,0% sampai 44,0%) | 2/50 = 4,0% (0,0% sampai 10,0%) |
| Fryum | 0,856 (0,794 sampai 0,908) | 0,882 | 0,742 | 30/100 = 30,0% (22,0% sampai 39,0%) | 5/50 = 10,0% (2,0% sampai 18,0%) |
| Permen karet | 0,860 (0,798 sampai 0,913) | 0,952 | 0,536 | 33/100 = 33,0% (24,0% sampai 42,0%) | 3/50 = 6,0% (0,0% sampai 14,0%) |
| Mete | 0,862 (0,804 sampai 0,916) | 0,754 | 0,669 | 48/100 = 48,0% (38,0% sampai 58,0%) | 1/50 = 2,0% (0,0% sampai 6,0%) |
| PCB 4 | 0,866 (0,815 sampai 0,911) | 0,894 | 0,588 | 46/100 = 46,0% (36,0% sampai 56,0%) | 6/101 = 5,9% (2,0% sampai 10,9%) |
| PCB 2 | 0,922 (0,883 sampai 0,957) | 0,947 | 0,834 | 32/100 = 32,0% (23,0% sampai 40,0%) | 1/100 = 1,0% (0,0% sampai 3,0%) |
| PCB 1 | 0,930 (0,887 sampai 0,963) | 0,984 | 0,827 | 21/100 = 21,0% (13,0% sampai 29,0%) | 7/100 = 7,0% (2,0% sampai 12,0%) |

Menurut ukuran cacat (bagian luas foto):

| Ukuran | Foto | Tertangkap | mIoU |
|---|---|---|---|
| kecil (< 0,2%) | 619 | 36,8% | 0,054 |
| sedang (0,2 sampai 1%) | 388 | 67,8% | 0,175 |
| besar (> 1%) | 193 | 83,4% | 0,250 |

## PatchCore ResNet18 (model browser)

| Produk | Image AUROC (CI 95%) | Pixel AUROC | AUPRO | Cacat lolos (CI 95%) | Normal ditolak (CI 95%) |
|---|---|---|---|---|---|
| Kapsul | 0,731 (0,650 sampai 0,805) | 0,972 | 0,705 | 44/100 = 44,0% (34,0% sampai 54,0%) | 10/60 = 16,7% (8,3% sampai 26,7%) |
| Makaroni 2 | 0,772 (0,700 sampai 0,834) | 0,950 | 0,790 | 60/100 = 60,0% (51,0% sampai 69,0%) | 12/100 = 12,0% (6,0% sampai 18,0%) |
| Fryum | 0,921 (0,878 sampai 0,957) | 0,952 | 0,815 | 25/100 = 25,0% (17,0% sampai 34,0%) | 2/50 = 4,0% (0,0% sampai 10,0%) |
| PCB 3 | 0,923 (0,885 sampai 0,955) | 0,986 | 0,858 | 33/100 = 33,0% (24,0% sampai 42,0%) | 5/101 = 5,0% (1,0% sampai 8,9%) |
| Makaroni 1 | 0,930 (0,888 sampai 0,964) | 0,985 | 0,910 | 29/100 = 29,0% (20,0% sampai 38,0%) | 4/100 = 4,0% (0,0% sampai 8,0%) |
| PCB 2 | 0,937 (0,898 sampai 0,969) | 0,981 | 0,849 | 16/100 = 16,0% (9,0% sampai 23,0%) | 7/100 = 7,0% (2,0% sampai 12,0%) |
| PCB 1 | 0,945 (0,912 sampai 0,970) | 0,995 | 0,867 | 32/100 = 32,0% (23,0% sampai 42,0%) | 4/100 = 4,0% (1,0% sampai 8,0%) |
| Mete | 0,953 (0,920 sampai 0,979) | 0,990 | 0,925 | 16/100 = 16,0% (9,0% sampai 22,0%) | 4/50 = 8,0% (2,0% sampai 16,0%) |
| Lilin | 0,981 (0,966 sampai 0,992) | 0,987 | 0,941 | 7/100 = 7,0% (3,0% sampai 12,0%) | 12/100 = 12,0% (6,0% sampai 19,0%) |
| PCB 4 | 0,992 (0,983 sampai 0,998) | 0,983 | 0,872 | 3/100 = 3,0% (0,0% sampai 7,0%) | 7/101 = 6,9% (3,0% sampai 11,9%) |
| Fryum pipa | 0,996 (0,989 sampai 1,000) | 0,991 | 0,937 | 1/100 = 1,0% (0,0% sampai 4,0%) | 5/50 = 10,0% (2,0% sampai 20,0%) |
| Permen karet | 0,997 (0,990 sampai 1,000) | 0,986 | 0,830 | 3/100 = 3,0% (0,0% sampai 7,0%) | 1/50 = 2,0% (0,0% sampai 6,0%) |

Menurut ukuran cacat (bagian luas foto):

| Ukuran | Foto | Tertangkap | mIoU |
|---|---|---|---|
| kecil (< 0,2%) | 619 | 63,3% | 0,038 |
| sedang (0,2 sampai 1%) | 388 | 89,7% | 0,135 |
| besar (> 1%) | 193 | 99,0% | 0,345 |

## PatchCore WideResNet50

| Produk | Image AUROC (CI 95%) | Pixel AUROC | AUPRO | Cacat lolos (CI 95%) | Normal ditolak (CI 95%) |
|---|---|---|---|---|---|
| Makaroni 2 | 0,622 (0,543 sampai 0,698) | 0,954 | 0,802 | 85/100 = 85,0% (78,0% sampai 92,0%) | 7/100 = 7,0% (2,0% sampai 12,0%) |
| Kapsul | 0,748 (0,673 sampai 0,820) | 0,968 | 0,682 | 57/100 = 57,0% (47,0% sampai 67,0%) | 1/60 = 1,7% (0,0% sampai 5,0%) |
| Makaroni 1 | 0,897 (0,856 sampai 0,937) | 0,978 | 0,894 | 47/100 = 47,0% (38,0% sampai 57,0%) | 5/100 = 5,0% (1,0% sampai 10,0%) |
| Fryum | 0,938 (0,900 sampai 0,968) | 0,954 | 0,795 | 27/100 = 27,0% (19,0% sampai 36,0%) | 0/50 = 0,0% (0,0% sampai 0,0%) |
| PCB 2 | 0,954 (0,923 sampai 0,982) | 0,985 | 0,884 | 18/100 = 18,0% (11,0% sampai 26,0%) | 5/100 = 5,0% (1,0% sampai 9,0%) |
| PCB 1 | 0,955 (0,926 sampai 0,978) | 0,997 | 0,924 | 8/100 = 8,0% (3,0% sampai 14,0%) | 10/100 = 10,0% (4,0% sampai 17,0%) |
| PCB 3 | 0,955 (0,929 sampai 0,977) | 0,987 | 0,852 | 21/100 = 21,0% (14,0% sampai 29,0%) | 8/101 = 7,9% (3,0% sampai 13,9%) |
| Mete | 0,974 (0,949 sampai 0,991) | 0,993 | 0,930 | 4/100 = 4,0% (1,0% sampai 8,0%) | 9/50 = 18,0% (8,0% sampai 28,0%) |
| Lilin | 0,979 (0,962 sampai 0,992) | 0,993 | 0,955 | 3/100 = 3,0% (0,0% sampai 7,0%) | 16/100 = 16,0% (9,0% sampai 23,0%) |
| Fryum pipa | 0,989 (0,977 sampai 0,998) | 0,994 | 0,935 | 2/100 = 2,0% (0,0% sampai 5,0%) | 5/50 = 10,0% (2,0% sampai 18,0%) |
| PCB 4 | 0,990 (0,980 sampai 0,997) | 0,984 | 0,881 | 3/100 = 3,0% (0,0% sampai 6,0%) | 12/101 = 11,9% (5,9% sampai 18,8%) |
| Permen karet | 0,996 (0,989 sampai 1,000) | 0,984 | 0,845 | 5/100 = 5,0% (1,0% sampai 10,0%) | 1/50 = 2,0% (0,0% sampai 6,0%) |

Menurut ukuran cacat (bagian luas foto):

| Ukuran | Foto | Tertangkap | mIoU |
|---|---|---|---|
| kecil (< 0,2%) | 619 | 58,8% | 0,044 |
| sedang (0,2 sampai 1%) | 388 | 94,1% | 0,137 |
| besar (> 1%) | 193 | 99,0% | 0,318 |

## Model browser (ONNX int8)

```
{
 "image_auroc_torch": 0.9232658140814082,
 "image_auroc_int8": 0.8126017051705171,
 "pixel_auroc_torch": 0.9797977655610569,
 "pixel_auroc_int8": 0.9609191753896423
}
```
