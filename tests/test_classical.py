import numpy as np
import pytest

from defect_lab import classical


def test_lbp_uniform_flat_and_spot():
    flat = np.full((5, 5), 100, dtype=np.uint8)
    # Semua tetangga >= pusat -> 8 bit satu -> pola uniform 8.
    assert (classical.lbp_uniform(flat) == 8).all()
    spot = flat.copy()
    spot[2, 2] = 200
    # Pusat lebih terang dari semua tetangga -> 0 bit satu -> pola 0.
    assert classical.lbp_uniform(spot)[2, 2] == 0


def test_describe_shape_and_determinism():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    a = classical.describe(img)
    assert a.shape == (1024, classical.DIM)
    assert a.dtype == np.float32
    assert np.array_equal(a, classical.describe(img))
    # Histogram LBP per patch berjumlah 1.
    assert np.allclose(a[:, -10:].sum(1), 1.0, atol=1e-4)


def test_describe_localises_a_spot():
    img = np.full((256, 256, 3), 128, dtype=np.uint8)
    base = classical.describe(img)
    img[100:110, 180:190] = (250, 20, 20)
    feat = classical.describe(img)
    diff = np.abs(feat - base).sum(1).reshape(32, 32)
    r, c = np.unravel_index(np.argmax(diff), diff.shape)
    # Pusat patch ke-k ada di piksel 8k+4; noda di baris 100-110, kolom 180-190.
    assert abs(8 * r + 4 - 105) <= 8 and abs(8 * c + 4 - 185) <= 8


def test_describe_rejects_wrong_size():
    with pytest.raises(ValueError):
        classical.describe(np.zeros((128, 128, 3), dtype=np.uint8))
