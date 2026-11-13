"""Train an anomaly model (PatchCore by default) on dataset/.

Training uses dataset/training/good only; dataset/test/{good,marker} are for evaluation.

Example:
    python vision/train.py --model patchcore
"""

import argparse
from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore, Stfpm

MODELS = {"patchcore": Patchcore, "padim": Padim, "stfpm": Stfpm}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("dataset"))
    p.add_argument("--name", default="bricks")
    p.add_argument("--normal-dir", default="training/good")
    p.add_argument("--normal-test-dir", default="test/good")
    p.add_argument("--abnormal-dir", default="test/marker")
    p.add_argument("--model", choices=MODELS, default="patchcore")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--accelerator", default="cpu", help="cpu (default; MPS crashes on macOS) | auto | gpu")
    p.add_argument("--results", type=Path, default=Path("results"), help="output root (gitignored)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dm = Folder(
        name=args.name,
        root=args.root,
        normal_dir=args.normal_dir,
        normal_test_dir=args.normal_test_dir,
        abnormal_dir=args.abnormal_dir,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=42,
    )
    model = MODELS[args.model]()
    engine = Engine(default_root_dir=args.results, accelerator=args.accelerator)
    engine.fit(datamodule=dm, model=model)
    print("\nTest metrics:")
    engine.test(datamodule=dm, model=model)
    print(f"\nCheckpoint: {engine.trainer.checkpoint_callback.best_model_path or 'see ' + str(args.results)}")


if __name__ == "__main__":
    main()
