"""Baseline klasik tanpa deep learning: deskriptor tangan per patch dengan OpenCV.

Setiap citra 256 x 256 dibagi menjadi grid 32 x 32 patch (langkah 8 piksel, jendela 16).
Per patch dihitung 24 angka:
  - rata-rata dan simpangan baku warna Lab (6),
  - histogram orientasi gradien Sobel berbobot magnitudo, 8 arah (8),
  - histogram Local Binary Pattern uniform 8 tetangga, 10 pola (10).
Skor anomali patch = jarak ke patch normal terdekat (kNN), sama seperti PatchCore, sehingga
perbandingan dengan transfer learning hanya berbeda pada fiturnya.
"""

from __future__ import annotations

import cv2
import numpy as np

GRID = 32
STRIDE = 8
WINDOW = 16
DIM = 24

# Peta kode LBP 8 bit -> 10 kelas uniform (0..8 = jumlah bit 1 pada pola uniform, 9 = lainnya).
_LBP_MAP = np.full(256, 9, dtype=np.uint8)
for _code in range(256):
    _bits = [(_code >> i) & 1 for i in range(8)]
    _transitions = sum(_bits[i] != _bits[(i + 1) % 8] for i in range(8))
    if _transitions <= 2:
        _LBP_MAP[_code] = sum(_bits)


def lbp_uniform(gray: np.ndarray) -> np.ndarray:
    """LBP radius 1, 8 tetangga, dipetakan ke 10 pola uniform. Tepi memakai replikasi."""
    g = np.pad(gray, 1, mode="edge").astype(np.int16)
    c = g[1:-1, 1:-1]
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    code = np.zeros(c.shape, dtype=np.uint8)
    h, w = c.shape
    for bit, (dy, dx) in enumerate(offsets):
        nb = g[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        code |= ((nb >= c).astype(np.uint8) << bit).astype(np.uint8)
    return _LBP_MAP[code]


def _pool(ch: np.ndarray) -> np.ndarray:
    """Rata-rata jendela 16 x 16, diambil di pusat grid 32 x 32."""
    box = cv2.boxFilter(ch, cv2.CV_32F, (WINDOW, WINDOW), borderType=cv2.BORDER_REFLECT)
    centers = np.arange(STRIDE // 2, STRIDE * GRID, STRIDE)
    return box[np.ix_(centers, centers)]


def describe(rgb: np.ndarray) -> np.ndarray:
    """Deskriptor [GRID*GRID, 24] float32 untuk satu citra RGB uint8 256 x 256."""
    if rgb.shape[:2] != (GRID * STRIDE, GRID * STRIDE):
        raise ValueError("citra harus 256 x 256")
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gf = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy)
    orient = np.floor((ang % np.pi) / (np.pi / 8)).astype(np.int64).clip(0, 7)
    lbp = lbp_uniform(gray)

    feats = []
    for k in range(3):
        m = _pool(lab[..., k])
        sq = _pool(lab[..., k] ** 2)
        feats.append(m)
        feats.append(np.sqrt(np.maximum(sq - m**2, 0.0)))
    # Urutan: mean L, std L, mean a, std a, mean b, std b.
    for k in range(8):
        feats.append(np.log1p(10.0 * _pool(mag * (orient == k))))
    for k in range(10):
        feats.append(_pool((lbp == k).astype(np.float32)))
    out = np.stack(feats, axis=-1).reshape(GRID * GRID, DIM)
    return out.astype(np.float32)


def describe_batch(images: np.ndarray) -> np.ndarray:
    return np.stack([describe(im) for im in images])
