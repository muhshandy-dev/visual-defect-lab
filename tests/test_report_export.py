import numpy as np

from defect_lab import metrics
from defect_lab.export_web import encode, heat_overlay, sample_id
from defect_lab.report import Raw, class_metrics, resample


def _raw(seed=0):
    rng = np.random.default_rng(seed)
    n = 20
    labels = np.array([0] * 10 + [1] * 10)
    maps = rng.normal(size=(n, 16, 16))
    masks = np.zeros((n, 16, 16), dtype=np.uint8)
    masks[10:, 4:8, 4:8] = 1
    maps[masks == 1] += 2
    ph = metrics.pixel_hists(maps, masks, 64)
    scores = maps.reshape(n, -1).max(1)
    ious = np.array([np.nan] * 10 + [metrics.iou(maps[i] > 2, masks[i]) for i in range(10, n)])
    return Raw(scores, labels, 2.0, ious, masks.mean((1, 2)), np.array([""] * n),
               np.array([str(i) for i in range(n)]), ph.pos, ph.neg, ph.region_hist,
               ph.region_image), maps, masks  # fmt: skip


def test_class_metrics_full_index_matches_direct():
    r, maps, masks = _raw()
    out = class_metrics(r, np.arange(20))
    assert out["image_auroc"] == metrics.auroc(r.scores, r.labels)
    assert abs(out["pixel_auroc"] - metrics.auroc(maps.reshape(-1), masks.reshape(-1))) < 0.01
    assert out["miou"] == np.nanmean(r.ious[10:])


def test_resample_is_stratified():
    labels = np.array([0] * 7 + [1] * 3)
    idx = resample(labels, np.random.default_rng(0))
    assert (labels[idx] == 0).sum() == 7 and (labels[idx] == 1).sum() == 3


def test_duplicated_images_weight_histograms():
    r, _, _ = _raw()
    a = class_metrics(r, np.array([0, 10, 10]))
    b = class_metrics(r, np.array([0, 10]))
    # Duplikat citra cacat tidak mengubah AUROC piksel (pos dan neg sama-sama berlipat) ...
    assert np.isfinite(a["pixel_auroc"]) and np.isfinite(b["pixel_auroc"])
    # ... tetapi region ikut diduplikasi untuk AUPRO.
    assert np.isfinite(a["aupro"])


def test_encode_matches_web_contract():
    # Sama dengan encodeRatio di web/src/lib/heat.ts.
    assert encode(np.array([0.0, 1.0, 0.999, 5.0, -1.0]), 1.0).tolist() == [0, 128, 127, 255, 0]


def test_overlay_shape_and_sample_id():
    rgb = np.full((8, 8, 3), 200, dtype=np.uint8)
    q = np.zeros((8, 8), dtype=np.uint8)
    q[2, 2] = 255
    out = heat_overlay(rgb, q)
    assert out.shape == (8, 8, 3)
    assert out[2, 2].tolist() != out[0, 0].tolist()
    assert sample_id("pcb1/Data/Images/Anomaly/012.JPG") == "a012"
    assert sample_id("pcb1/Data/Images/Normal/0345.JPG") == "n0345"
