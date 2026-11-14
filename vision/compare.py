"""Compare anomaly models: train each, test, export to OpenVINO, benchmark latency.

Produces one table (accuracy vs latency vs size vs training time) - the slide for
"which model and why". Run the final comparison on the station laptop.

Example:
    python vision/compare.py --models patchcore padim stfpm --devices CPU
    python vision/compare.py --root dataset_synth --precision int8   # on synthetic defects
"""

import argparse
import csv
import time
from pathlib import Path

from anomalib.engine import Engine

from benchmark import bench, model_size_mb
from common import MODELS, list_test_set, make_datamodule
from export import export


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", choices=MODELS, default=["patchcore", "padim", "stfpm"])
    p.add_argument("--root", type=Path, default=Path("dataset/processed"))
    p.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    p.add_argument("--devices", nargs="+", default=["CPU"])
    p.add_argument("--epochs", type=int, default=None, help="override max epochs for trainable models")
    p.add_argument("--accelerator", default="cpu", help="training device (cpu; MPS crashes on macOS)")
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--out", type=Path, default=Path("results/compare"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    test_set = list_test_set(args.root)
    rows = []
    for name in args.models:
        cls, epochs = MODELS[name]
        if args.epochs and epochs > 1:
            epochs = args.epochs
        print(f"\n=== {name} ({epochs} epoch{'s' if epochs > 1 else ''}) ===")
        batch = 1 if name == "efficient_ad" else 8  # EfficientAd requires batch size 1
        dm = make_datamodule(args.root, batch_size=batch)
        model = cls()
        engine = Engine(default_root_dir=args.out / "train", accelerator=args.accelerator,
                        max_epochs=epochs, logger=False)
        t0 = time.perf_counter()
        engine.fit(datamodule=dm, model=model)
        train_s = time.perf_counter() - t0
        ckpt = Path(engine.trainer.checkpoint_callback.best_model_path)

        xml = export(ckpt, name, [args.precision], args.root, args.out / "export")[args.precision]
        for dev in args.devices:
            try:
                r = bench(xml, dev, test_set, args.runs, warmup=3)
            except Exception as e:
                print(f"{name} on {dev} FAILED: {str(e).splitlines()[0][:120]}")
                continue
            rows.append({"model": name, "precision": args.precision, "device": dev, "train_s": train_s,
                         "size_mb": model_size_mb(xml), **r})

    if not rows:
        return
    print(f"\n{'model':14s} {'dev':4s} {'AUROC':>6s} {'F1':>6s} {'p50 ms':>8s} {'p95 ms':>8s} "
          f"{'fps':>6s} {'MB':>6s} {'train s':>8s}")
    for r in rows:
        print(f"{r['model']:14s} {r['device']:4s} {r['auroc']:6.3f} {r['f1']:6.3f} {r['p50_ms']:8.1f} "
              f"{r['p95_ms']:8.1f} {r['fps']:6.1f} {r['size_mb']:6.0f} {r['train_s']:8.0f}")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "compare.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {args.out / 'compare.csv'}")


if __name__ == "__main__":
    main()
