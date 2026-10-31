"""Unduh, verifikasi, indeks, dan cache VisA.

Split yang dipakai adalah split resmi `split_csv/1cls.csv` dari VisA. Citra normal train
dibagi lagi secara acak berbenih menjadi fit (masuk memory bank) dan validasi (hanya
untuk ambang operasi). Data test tidak pernah dipakai untuk fit, ambang, atau pemilihan
hiperparameter.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tarfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from . import config


@dataclass
class Item:
    cls: str
    split: str  # fit, val, test
    label: int  # 0 normal, 1 cacat
    image: str  # path relatif terhadap akar VisA
    mask: str
    defect: str  # jenis cacat menurut image_anno.csv, "" untuk normal


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(raw_dir: Path = config.RAW_DIR) -> Path:
    """Unduh tar VisA (sekali), cek SHA-256, lalu ekstrak. Mengembalikan akar VisA."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    tar_path = raw_dir / "VisA_20220922.tar"
    root = raw_dir / "visa"
    if not tar_path.exists() or tar_path.stat().st_size != config.VISA_BYTES:
        tmp = tar_path.with_suffix(".part")
        with urllib.request.urlopen(config.VISA_URL) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        tmp.rename(tar_path)
    digest = sha256_file(tar_path)
    if digest != config.VISA_SHA256:
        raise RuntimeError(f"SHA-256 VisA tidak cocok: {digest}")
    if not (root / "split_csv" / "1cls.csv").exists():
        root.mkdir(exist_ok=True)
        with tarfile.open(tar_path) as t:
            t.extractall(root, filter="data")
    return root


def build_index(root: Path, seed: int = config.SEED) -> list[Item]:
    """Baca split resmi lalu sisihkan validasi dari citra normal train per kelas."""
    rows = list(csv.DictReader((root / "split_csv" / "1cls.csv").open()))
    defects: dict[str, str] = {}
    for cls in config.CLASSES:
        for r in csv.DictReader((root / cls / "image_anno.csv").open()):
            defects[r["image"]] = "" if r["label"] == "normal" else r["label"]
    items: list[Item] = []
    for cls in config.CLASSES:
        train = sorted(r["image"] for r in rows if r["object"] == cls and r["split"] == "train")
        rng = np.random.default_rng([seed, config.CLASSES.index(cls)])
        n_val = int(round(len(train) * config.VAL_FRACTION))
        val = set(rng.choice(train, size=n_val, replace=False).tolist())
        for img in train:
            items.append(Item(cls, "val" if img in val else "fit", 0, img, "", ""))
        test = sorted(
            (r for r in rows if r["object"] == cls and r["split"] == "test"),
            key=lambda r: r["image"],
        )
        for r in test:
            lab = int(r["label"] == "anomaly")
            items.append(Item(cls, "test", lab, r["image"], r["mask"], defects.get(r["image"], "")))
    return items


def write_split(items: list[Item], path: Path) -> None:
    """Simpan pembagian data agar bisa diperiksa tanpa mengunduh VisA."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "split", "label", "image", "defect"])
        for it in items:
            w.writerow([it.cls, it.split, it.label, it.image, it.defect])


def load_image(path: Path, size: int = config.IMAGE_SIZE) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def load_mask(path: Path, size: int = config.IMAGE_SIZE) -> np.ndarray:
    """Masker biner. Resize nearest, kecuali cacat kecil hilang: pakai INTER_AREA > 0."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    m = (m > 0).astype(np.uint8)
    out = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    if m.any() and not out.any():
        area = cv2.resize(m.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
        out = (area > 0).astype(np.uint8)
    return out


def cache_path(cls: str, size: int = config.IMAGE_SIZE) -> Path:
    return config.CACHE_DIR / str(size) / f"{cls}.npz"


def build_cache(root: Path, items: list[Item], size: int = config.IMAGE_SIZE) -> None:
    for cls in config.CLASSES:
        out = cache_path(cls, size)
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        sub = [it for it in items if it.cls == cls]
        imgs = np.stack([load_image(root / it.image, size) for it in sub])
        masks = np.zeros((len(sub), size, size), dtype=np.uint8)
        for i, it in enumerate(sub):
            if it.label:
                masks[i] = load_mask(root / it.mask, size)
        meta = json.dumps([asdict(it) for it in sub])
        np.savez(out, images=imgs, masks=masks, meta=np.array(meta))


@dataclass
class ClassData:
    cls: str
    images: np.ndarray  # [N, S, S, 3] uint8 RGB
    masks: np.ndarray  # [N, S, S] uint8
    items: list[Item]

    def select(self, split: str) -> tuple[np.ndarray, np.ndarray, list[Item]]:
        idx = [i for i, it in enumerate(self.items) if it.split == split]
        return self.images[idx], self.masks[idx], [self.items[i] for i in idx]


def load_class(cls: str, size: int = config.IMAGE_SIZE) -> ClassData:
    z = np.load(cache_path(cls, size))
    items = [Item(**d) for d in json.loads(str(z["meta"]))]
    return ClassData(cls, z["images"], z["masks"], items)
