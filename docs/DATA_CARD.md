# Data card: VisA untuk Visual Defect Lab

## Sumber dan lisensi

- Dataset: VisA (Visual Anomaly), Amazon, arsip `VisA_20220922.tar` dari `https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar`.
- Ukuran 1.929.840.640 byte, SHA-256 `2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362` (diperiksa oleh `defect-lab data`).
- Lisensi CC BY 4.0 (berkas `LICENSE-DATASET` di dalam arsip). Atribusi dan daftar perubahan ada di `NOTICE.md` dan footer website.

## Isi

12 produk: lilin (candle), kapsul, kacang mete, permen karet, kerupuk fryum, makaroni 1 dan 2, PCB 1 sampai 4, kerupuk pipa. Semua foto RGB, sekitar 1.000 sampai 1.500 piksel per sisi. Setiap foto cacat punya masker biner area cacat dan label jenis cacat berbahasa Inggris (misalnya "melt", "small scratches"; diterjemahkan di website).

## Pembagian

Dipakai split resmi `split_csv/1cls.csv` (satu model per produk, latih hanya dari foto normal).

| Bagian | Isi | Jumlah | Dipakai untuk |
|---|---|---|---|
| fit | 90% foto latih resmi (normal) | 7.795 | memori patch normal (memory bank) |
| val | 10% foto latih resmi (normal), acak berbenih 2026 per produk | 864 | ambang operasi saja |
| test | foto uji resmi | 962 normal + 1.200 cacat | evaluasi, sekali per konfigurasi |

Daftar lengkap per foto ada di `evaluation/split.csv` dan diperiksa oleh `tools/verify_published.py` (tidak ada tumpang tindih, semua foto cacat di test, val tepat 10%).

## Praproses

- Foto diperkecil ke 256 x 256 dengan `cv2.INTER_AREA` tanpa dipotong (rasio aspek tidak dipertahankan).
- Masker diperkecil dengan nearest neighbour; bila cacat kecil hilang total, dipakai `INTER_AREA > 0` agar setiap foto cacat tetap punya area cacat.
- Luas cacat (bagian dari foto) dihitung pada masker 256 x 256. Distribusi: 619 foto di bawah 0,2%, 388 antara 0,2% dan 1%, 193 di atas 1%.

## Batasan data

- Satu kamera dan pencahayaan per produk. Foto dari pabrik lain hampir pasti berbeda.
- Jumlah foto normal uji per produk kecil (50 sampai 101), sehingga tingkat alarm palsu per produk tidak presisi.
- Beberapa label jenis cacat ditulis tidak seragam di dataset asli ("different colour spot" dan "different color spot"); keduanya diperlakukan sebagai label berbeda di analisis per jenis cacat, tetapi diterjemahkan sama di website.
