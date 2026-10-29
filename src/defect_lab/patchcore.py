"""PatchCore (Roth dkk., CVPR 2022) dengan backbone pralatih ImageNet yang dibekukan.

Transfer learning di sini berarti fitur tengah CNN pralatih (layer2 + layer3) dipakai apa
adanya tanpa fine-tuning. "Fit" = mengumpulkan fitur patch dari citra normal lalu memilih
coreset (k-center greedy) sebagai memory bank. Skor patch = jarak ke tetangga terdekat.

Kepala skor (`NearestHead`) dipakai bersama oleh baseline klasik, evaluasi Python, dan
ekspor ONNX, sehingga browser menjalankan rumus yang sama persis.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PatchEmbedder(nn.Module):
    """Citra [B,3,S,S] float 0..1 -> fitur patch [B, (S/8)^2, D]."""

    def __init__(self, backbone: str, embed_dim: int, weights: str | None = "auto"):
        """`weights`: "auto" = bobot torchvision resmi (diverifikasi SHA-256), None = acak."""
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            backbone, pretrained=False, features_only=True, out_indices=(2, 3)
        )
        if weights == "auto":
            load_torchvision_weights(self.backbone, backbone)
        self.embed_dim = embed_dim
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        f2, f3 = self.backbone(x)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3], dim=1)
        b, c, h, w = f.shape
        f = f.permute(0, 2, 3, 1).reshape(b, h * w, c)
        if c != self.embed_dim:
            f = F.adaptive_avg_pool1d(f, self.embed_dim)
        return f


def weights_path(backbone: str) -> Path:
    return config.WEIGHTS_DIR / config.TORCHVISION_WEIGHTS[backbone][0]


def load_torchvision_weights(model: nn.Module, backbone: str) -> None:
    """Muat bobot ImageNet torchvision. Nama berkas memuat 8 heksa pertama SHA-256."""
    from .data import sha256_file

    fname, sha = config.TORCHVISION_WEIGHTS[backbone]
    path = weights_path(backbone)
    if not path.exists():
        raise FileNotFoundError(
            f"Bobot {fname} tidak ada di {config.WEIGHTS_DIR}. Unduh dari "
            f"https://download.pytorch.org/models/{fname}"
        )
    digest = sha256_file(path)
    if not digest.startswith(sha):
        raise RuntimeError(f"SHA-256 {fname} tidak cocok: {digest}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = {k: v for k, v in state.items() if not k.startswith(("fc.", "layer4."))}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith("layer4.")]
    if missing or unexpected:
        raise RuntimeError(f"bobot tidak cocok: hilang={missing} lebih={unexpected}")


def to_tensor(images: np.ndarray) -> torch.Tensor:
    """uint8 [N,S,S,3] -> float [N,3,S,S] dalam 0..1."""
    return torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255.0)


@torch.no_grad()
def embed_images(model: PatchEmbedder, images: np.ndarray, batch: int = 16) -> torch.Tensor:
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(images), batch):
        out.append(model(to_tensor(images[i : i + batch]).to(device)).half().cpu())
    return torch.cat(out)


@torch.no_grad()
def greedy_coreset(x: torch.Tensor, m: int, proj_dim: int = 128, seed: int = 0) -> torch.Tensor:
    """k-center greedy pada proyeksi acak (Johnson-Lindenstrauss), seperti PatchCore."""
    n, d = x.shape
    if m >= n:
        return torch.arange(n)
    dev = torch.device(config.DEVICE)
    g = torch.Generator().manual_seed(seed)
    if proj_dim and proj_dim < d:
        w = (torch.randn(d, proj_dim, generator=g) / math.sqrt(proj_dim)).to(dev)
        z = torch.cat([x[i : i + 65536].to(dev).float() @ w for i in range(0, n, 65536)])
    else:
        z = x.to(dev).float().clone()
    zn = (z * z).sum(1)
    start = int(torch.randint(n, (1,), generator=g))
    idx = torch.empty(m, dtype=torch.long)
    idx[0] = start
    mind = (zn - 2 * (z @ z[start]) + zn[start]).clamp_min_(0)
    for k in range(1, m):
        i = int(torch.argmax(mind))
        idx[k] = i
        torch.minimum(mind, (zn - 2 * (z @ z[i]) + zn[i]).clamp_min_(0), out=mind)
    return idx


def gaussian_kernel(sigma: float) -> torch.Tensor:
    radius = int(4 * sigma + 0.5)
    t = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum()


class NearestHead(nn.Module):
    """Fitur patch [B,P,D] -> (peta skor [B,S,S], skor patch [B,P]).

    Jarak Euclid ke tetangga terdekat di bank, lalu interpolasi bilinear ke S x S dan
    penghalusan Gaussian (pad reflect). Bank disimpan float16 bila `half_bank`.
    """

    def __init__(
        self,
        bank: torch.Tensor,
        grid: int = 32,
        size: int = config.IMAGE_SIZE,
        sigma: float = config.SMOOTH_SIGMA,
        half_bank: bool = False,
    ):
        super().__init__()
        bank = bank.float()
        self.half_bank = half_bank
        self.register_buffer("bank", bank.half() if half_bank else bank)
        stored = self.bank.float()
        self.register_buffer("bank_sq", (stored * stored).sum(1))
        k = gaussian_kernel(sigma)
        self.register_buffer("kx", k.view(1, 1, 1, -1))
        self.register_buffer("ky", k.view(1, 1, -1, 1))
        self.pad = (k.numel() - 1) // 2
        self.grid = grid
        self.size = size

    def forward(self, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bank = self.bank.float()
        e = emb.float()
        d2 = (e * e).sum(-1, keepdim=True) - 2.0 * torch.matmul(e, bank.t()) + self.bank_sq
        dmin = torch.sqrt(torch.clamp(d2.min(dim=-1).values, min=0.0))
        b = dmin.shape[0]
        patch = dmin.reshape(b, 1, self.grid, self.grid)
        up = F.interpolate(patch, size=(self.size, self.size), mode="bilinear", align_corners=False)
        p = self.pad
        up = F.conv2d(F.pad(up, (p, p, 0, 0), mode="reflect"), self.kx)
        up = F.conv2d(F.pad(up, (0, 0, p, p), mode="reflect"), self.ky)
        return up[:, 0], dmin


@torch.no_grad()
def score_embeddings(
    head: NearestHead, emb: torch.Tensor, batch: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    maps, patches = [], []
    for i in range(0, len(emb), batch):
        m, p = head(emb[i : i + batch].to(head.bank.device))
        maps.append(m.cpu().numpy())
        patches.append(p.cpu().numpy())
    return np.concatenate(maps), np.concatenate(patches)
