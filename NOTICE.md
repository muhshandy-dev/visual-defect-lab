# Atribusi pihak ketiga

## Dataset VisA

Foto, masker cacat, dan turunannya di `web/public/img`, `web/public/in`, `web/public/mask`, dan `web/public/map` berasal dari dataset **VisA (Visual Anomaly)**, Amazon, dirilis dengan lisensi [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

- Sumber: https://github.com/amazon-science/spot-diff, arsip `VisA_20220922.tar`.
- Sitasi: Zou, Y., Jeong, J., Pemula, L., Zhang, D., Dabeer, O. *SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation*. ECCV 2022.
- Perubahan: foto diperkecil (sisi terpanjang 720 piksel untuk tampilan, 256 x 256 untuk masukan model), dikompresi JPEG/PNG, dan masker diperkecil. Peta skor adalah keluaran model proyek ini, bukan bagian dari dataset.
- Tidak ada dukungan atau pengesahan dari pemilik dataset.

## Bobot model

Bobot ImageNet ResNet18 dan WideResNet50-2 dari torchvision (BSD-3-Clause). Model browser di `web/public/models` adalah turunan bobot ResNet18 itu yang dipotong sampai layer3 dan dikuantisasi int8, ditambah memori patch normal dari foto VisA.

## Font

IBM Plex Sans dan IBM Plex Mono (SIL Open Font License 1.1), dibundel dari paket @fontsource.
