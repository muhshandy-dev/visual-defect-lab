import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from defect_lab.patchcore import (  # noqa: E402
    NearestHead,
    PatchEmbedder,
    gaussian_kernel,
    greedy_coreset,
)

pytestmark = pytest.mark.model


def test_gaussian_kernel_normalised():
    k = gaussian_kernel(4.0)
    assert k.numel() == 33
    assert float(k.sum()) == pytest.approx(1.0)


def test_coreset_covers_clusters():
    g = torch.Generator().manual_seed(0)
    centers = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    x = torch.cat([c + 0.1 * torch.randn(200, 2, generator=g) for c in centers])
    idx = greedy_coreset(x, 4, proj_dim=0, seed=1)
    picked = x[idx]
    # k-center greedy dengan 4 titik harus mengambil satu titik dari tiap klaster.
    d = torch.cdist(picked, centers).argmin(1)
    assert sorted(d.tolist()) == [0, 1, 2, 3]
    assert len(set(idx.tolist())) == 4


def test_head_matches_brute_force_knn():
    g = torch.Generator().manual_seed(0)
    bank = torch.randn(300, 16, generator=g)
    emb = torch.randn(2, 64, 16, generator=g)
    head = NearestHead(bank, grid=8, size=32, sigma=1.0)
    maps, patch = head(emb)
    brute = torch.cdist(emb, bank.unsqueeze(0).expand(2, -1, -1)).min(-1).values
    assert torch.allclose(patch, brute, atol=1e-4)
    assert maps.shape == (2, 32, 32)
    # Penghalusan tidak mengubah rata-rata jauh dan tidak menghasilkan nilai di luar rentang.
    assert float(maps.max()) <= float(patch.max()) + 1e-5
    assert float(maps.min()) >= float(patch.min()) - 1e-5


def test_head_half_bank_rounds_only_bank():
    bank = torch.randn(50, 8)
    full = NearestHead(bank, grid=4, size=8, sigma=1.0)
    half = NearestHead(bank, grid=4, size=8, sigma=1.0, half_bank=True)
    assert half.bank.dtype == torch.float16
    emb = torch.randn(1, 16, 8)
    assert torch.allclose(full(emb)[1], half(emb)[1], atol=5e-3)


def test_embedder_shape_random_weights():
    m = PatchEmbedder("resnet18", 384, weights=None)
    with torch.no_grad():
        out = m(torch.rand(1, 3, 256, 256))
    assert out.shape == (1, 1024, 384)


def test_onnx_export_parity_random_weights(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    from defect_lab.export_onnx import export_backbone, export_head

    torch.manual_seed(0)
    m = PatchEmbedder("resnet18", 384, weights=None)
    head = NearestHead(torch.randn(256, 384), half_bank=True)
    export_backbone(m, tmp_path / "b.onnx")
    export_head(head, tmp_path / "h.onnx")
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        ref_map, _ = head(m(x))
    emb = ort.InferenceSession(str(tmp_path / "b.onnx")).run(None, {"image": x.numpy()})[0]
    out = ort.InferenceSession(str(tmp_path / "h.onnx")).run(None, {"patches": emb})[0]
    assert np.abs(out - ref_map.numpy()).max() < 1e-3


def test_weight_only_int8_is_small_and_faithful(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    from defect_lab.export_onnx import export_backbone, quantize_weights

    torch.manual_seed(0)
    m = PatchEmbedder("resnet18", 384, weights=None)
    export_backbone(m, tmp_path / "b.onnx")
    quantize_weights(tmp_path / "b.onnx", tmp_path / "q.onnx")
    assert (tmp_path / "q.onnx").stat().st_size < 0.35 * (tmp_path / "b.onnx").stat().st_size
    x = torch.rand(1, 3, 256, 256).numpy()
    a = ort.InferenceSession(str(tmp_path / "b.onnx")).run(None, {"image": x})[0]
    b = ort.InferenceSession(str(tmp_path / "q.onnx")).run(None, {"image": x})[0]
    # Hanya bobot yang dibulatkan: fitur hampir sama (relatif terhadap skala fitur).
    assert np.abs(a - b).max() < 0.02 * np.abs(a).max()
