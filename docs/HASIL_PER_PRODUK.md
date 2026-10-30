# Hasil per produk

Berkas ini dibuat otomatis oleh `tools/render_report.py` dari `evaluation/results.json`. Jangan diedit tangan.

Semua angka dihitung pada data uji resmi VisA (split 1cls). CI 95% berasal dari bootstrap bertingkat 1000 kali (foto normal dan foto cacat diambil ulang terpisah di tiap produk). Ambang operasi tiap produk ditetapkan dari kuantil 0,95 skor foto normal validasi, tidak pernah dari data uji, jadi laju cacat lolos dan normal ditolak adalah angka jujur untuk ambang yang benar-benar akan dipakai.

## Rata-rata makro 12 produk

| Metrik | Klasik OpenCV (LBP + gradien + warna, kNN) |
|---|---|
| Image AUROC | 0,811 (0,793 sampai 0,828) |
| Pixel AUROC | 0,915 (0,907 sampai 0,924) |
| AUPRO (FPR 0,3) | 0,716 (0,698 sampai 0,737) |
| mIoU | 0,125 (0,115 sampai 0,134) |
| Normal ditolak | 7,1% (5,5% sampai 8,9%) |
| Cacat lolos | 45,7% (43,1% sampai 48,3%) |

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
| Permen karet | 0,859 (0,798 sampai 0,913) | 0,952 | 0,536 | 33/100 = 33,0% (24,0% sampai 42,0%) | 3/50 = 6,0% (0,0% sampai 14,0%) |
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
