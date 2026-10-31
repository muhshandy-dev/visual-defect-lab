"""Tulis docs/HASIL_PER_PRODUK.md langsung dari evaluation/results.json.

Semua angka di berkas itu berasal dari skrip ini, bukan diketik tangan, sehingga laporan
selalu sama dengan hasil evaluasi. Jalankan ulang setiap kali results.json berubah.

Pemakaian: python tools/render_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evaluation" / "results.json"
PARITY = ROOT / "evaluation" / "onnx_parity.json"
OUT = ROOT / "docs" / "HASIL_PER_PRODUK.md"

NAMES = {
    "candle": "Lilin", "capsules": "Kapsul", "cashew": "Mete", "chewinggum": "Permen karet",
    "fryum": "Fryum", "macaroni1": "Makaroni 1", "macaroni2": "Makaroni 2", "pcb1": "PCB 1",
    "pcb2": "PCB 2", "pcb3": "PCB 3", "pcb4": "PCB 4", "pipe_fryum": "Fryum pipa",
}  # fmt: skip
METRIC_NAMES = {
    "image_auroc": "Image AUROC", "pixel_auroc": "Pixel AUROC", "aupro": "AUPRO (FPR 0,3)",
    "miou": "mIoU", "false_reject": "Normal ditolak", "miss": "Cacat lolos",
}  # fmt: skip


def f(v: float, d: int = 3) -> str:
    return f"{v:.{d}f}".replace(".", ",")


def p(v: float, d: int = 1) -> str:
    return f"{v * 100:.{d}f}%".replace(".", ",")


def pp(v: float) -> str:
    """Selisih laju dalam poin persen."""
    return f"{v * 100:.1f} poin".replace(".", ",")


def ci(c: list[float], d: int = 3) -> str:
    return f"{f(c[0], d)} sampai {f(c[1], d)}"


def pci(c: list[float]) -> str:
    return f"{p(c[0])} sampai {p(c[1])}"


def onnx_section(parity: dict) -> list[str]:
    bb = parity["backbone"]
    mb = 1024 * 1024
    out = [
        "",
        "## Model browser (PatchCore ResNet18, ONNX)",
        "",
        f"Backbone {bb['bytes'] / mb:.1f} MB (float32 {bb['fp32_bytes'] / mb:.1f} MB), varian "
        f"kuantisasi `{bb.get('variant', 'static')}`.".replace(".", ",", 2),
    ]
    sel = parity.get("selection")
    if sel:
        out += [
            "",
            f"Aturan pemilihan: {sel['rule']}. Hanya foto normal validasi yang dipakai.",
            "",
            "| Varian | Selisih maks / ambang (validasi) | Ukuran | Lolos aturan |",
            "|---|---|---|---|",
        ]
        for v, d in sel["val_max_rel_diff"].items():
            ok = "ya" if d <= sel["fidelity_limit"] else "tidak"
            out.append(
                f"| {v} | {f(d)} | {sel['bytes'][v] / mb:.2f} MB | {ok} |".replace(".", ",", 1)
            )
    m = parity["macro"]
    out += [
        "",
        f"Pada seluruh {parity['test_images_total']} foto uji: image AUROC makro PyTorch "
        f"{f(m['image_auroc_torch'])}, ONNX {f(m['image_auroc_int8'])}; pixel AUROC makro "
        f"PyTorch {f(m['pixel_auroc_torch'])}, ONNX {f(m['pixel_auroc_int8'])}. Keputusan lolos "
        f"atau tolak berubah pada {parity['decision_flips_int8_total']} foto.",
        "",
        "| Produk | Image AUROC PyTorch | Image AUROC ONNX | Keputusan berubah |",
        "|---|---|---|---|",
    ]
    for c, e in parity["classes"].items():
        auc = f"{f(e['image_auroc_torch'])} | {f(e['image_auroc_int8'])}"
        out.append(f"| {NAMES.get(c, c)} | {auc} | {e['decision_flips_int8']}/{e['test_images']} |")
    return out


def render(res: dict, parity: dict | None) -> str:
    methods = list(res["methods"])
    label = {m: res["methods"][m]["label"] for m in methods}
    reps = res["config"]["bootstrap_reps"]
    out = [
        "# Hasil per produk",
        "",
        "Berkas ini dibuat otomatis oleh `tools/render_report.py` dari `evaluation/results.json`. "
        "Jangan diedit tangan.",
        "",
        "Semua angka dihitung pada data uji resmi VisA (split 1cls). CI 95% berasal dari "
        f"bootstrap bertingkat {reps} kali (foto normal dan foto cacat diambil ulang terpisah "
        "di tiap produk). "
        "Ambang operasi tiap produk ditetapkan dari kuantil 0,95 skor foto normal validasi, "
        "tidak pernah dari data uji, jadi laju cacat lolos dan normal ditolak adalah angka jujur "
        "untuk ambang yang benar-benar akan dipakai.",
        "",
        "## Rata-rata makro 12 produk",
        "",
        "| Metrik | " + " | ".join(label[m] for m in methods) + " |",
        "|---|" + "---|" * len(methods),
    ]
    for k in ("image_auroc", "pixel_auroc", "aupro", "miou"):
        cells = [f"{f(res['methods'][m]['macro'][k])} ({ci(res['methods'][m]['macro_ci95'][k])})"
                 for m in methods]  # fmt: skip
        out.append(f"| {METRIC_NAMES[k]} | " + " | ".join(cells) + " |")
    for k in ("false_reject", "miss"):
        cells = []
        for m in methods:
            e = res["methods"][m]
            c = e["macro_ci95"].get(k)
            cells.append(f"{p(e[f'macro_{k}'])}" + (f" ({pci(c)})" if c else ""))
        out.append(f"| {METRIC_NAMES[k]} | " + " | ".join(cells) + " |")

    if res["paired"]:
        out += [
            "",
            "## Selisih berpasangan",
            "",
            "Selisih rata-rata makro dengan sampel bootstrap yang sama untuk kedua metode. "
            "Untuk AUROC, AUPRO dan mIoU, positif berarti lebih baik; untuk cacat lolos dan normal "
            "ditolak, negatif berarti lebih baik.",
            "",
            "| Perbandingan | Metrik | Selisih | CI 95% |",
            "|---|---|---|---|",
        ]
        for pair, diff in res["paired"].items():
            a, b = pair.split("_vs_")
            name = f"{label.get(a, a)} vs {label.get(b, b)}"
            for k, d in diff.items():
                rate = k in ("false_reject", "miss")
                val = pp(d["diff"]) if rate else f(d["diff"])
                band = f"{pp(d['ci95'][0])} sampai {pp(d['ci95'][1])}" if rate else ci(d["ci95"])
                out.append(f"| {name} | {METRIC_NAMES[k]} | {val} | {band} |")

    for m in methods:
        e = res["methods"][m]
        out += [
            "",
            f"## {label[m]}",
            "",
            "| Produk | Image AUROC (CI 95%) | Pixel AUROC | AUPRO | Cacat lolos (CI 95%) "
            "| Normal ditolak (CI 95%) |",
            "|---|---|---|---|---|---|",
        ]
        rows = sorted(e["per_class"].items(), key=lambda kv: kv[1]["image_auroc"])
        for c, r in rows:
            miss = p(r["test_miss"])
            fr = p(r["test_false_reject"])
            if "test_missed" in r:
                miss = (f"{r['test_missed']}/{r['test_defect_images']} = {miss} "
                        f"({pci(r['ci95']['miss'])})")  # fmt: skip
                fr = (f"{r['test_false_rejected']}/{r['test_normal_images']} = {fr} "
                      f"({pci(r['ci95']['false_reject'])})")  # fmt: skip
            out.append(
                f"| {NAMES.get(c, c)} | {f(r['image_auroc'])} ({ci(r['ci95']['image_auroc'])}) "
                f"| {f(r['pixel_auroc'])} | {f(r['aupro'])} | {miss} | {fr} |"
            )
        out += ["", "Menurut ukuran cacat (bagian luas foto):", "",
                "| Ukuran | Foto | Tertangkap | mIoU |", "|---|---|---|---|"]  # fmt: skip
        for row in e["by_size"]:
            out.append(f"| {row['group']} | {row['n']} | {p(row['detected'])} | {f(row['miou'])} |")

    if parity:
        out += onnx_section(parity)
    return "\n".join(out) + "\n"


def main() -> None:
    res = json.loads(RESULTS.read_text())
    parity = json.loads(PARITY.read_text()) if PARITY.exists() else None
    OUT.write_text(render(res, parity))
    print(f"ditulis {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
