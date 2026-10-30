"""CLI: defect-lab data | evaluate | report | export-onnx | export-web."""

from __future__ import annotations

import argparse
import json
import time

from . import config


def cmd_data(_: argparse.Namespace) -> None:
    from . import data

    root = data.fetch()
    items = data.build_index(root)
    data.build_cache(root, items)
    data.write_split(items, config.EVAL_DIR / "split.csv")
    counts: dict[str, int] = {}
    for it in items:
        key = f"{it.split}_{'cacat' if it.label else 'normal'}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps(counts, indent=2))


def cmd_evaluate(args: argparse.Namespace) -> None:
    import torch

    from .evaluate import load_embedder, run_class

    torch.set_num_threads(args.threads)
    method = config.METHODS[args.method]
    model = load_embedder(method)
    for cls in args.classes or config.CLASSES:
        t0 = time.perf_counter()
        s = run_class(method, model, cls)
        print(
            f"{method.key:9s} {cls:11s} img {s['image_auroc']:.3f} pix {s['pixel_auroc']:.3f} "
            f"pro {s['aupro']:.3f} miou {s['miou']:.3f} bank {s['bank_size']} "
            f"({time.perf_counter() - t0:.0f} s)",
            flush=True,
        )


def cmd_report(args: argparse.Namespace) -> None:
    from .report import build_results

    res = build_results(args.methods, reps=args.reps)
    out = config.EVAL_DIR / "results.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    for m, r in res["methods"].items():
        mac = r["macro"]
        print(m, {k: round(v, 4) for k, v in mac.items()})
    print(f"ditulis: {out}")


def cmd_export_onnx(_: argparse.Namespace) -> None:
    from .export_onnx import export_all

    print(json.dumps(export_all(), indent=2))


def cmd_export_web(args: argparse.Namespace) -> None:
    from .export_web import export, refresh_summary

    refresh_summary() if args.summary_only else export()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="defect-lab")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data", help="unduh VisA, cek SHA-256, buat cache 256 px").set_defaults(
        fn=cmd_data
    )
    e = sub.add_parser("evaluate", help="fit + evaluasi satu metode")
    e.add_argument("--method", choices=list(config.METHODS), required=True)
    e.add_argument("--classes", nargs="*")
    e.add_argument("--threads", type=int, default=4)
    e.set_defaults(fn=cmd_evaluate)
    r = sub.add_parser("report", help="gabungkan hasil + bootstrap")
    r.add_argument("--methods", nargs="+", default=list(config.METHODS))
    r.add_argument("--reps", type=int, default=config.BOOTSTRAP_REPS)
    r.set_defaults(fn=cmd_report)
    sub.add_parser("export-onnx", help="ekspor model browser ke ONNX int8").set_defaults(
        fn=cmd_export_onnx
    )
    ew = sub.add_parser("export-web", help="tulis data website")
    ew.add_argument("--summary-only", action="store_true",
                    help="hanya perbarui angka di summary.json dari results.json")  # fmt: skip
    ew.set_defaults(fn=cmd_export_web)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
