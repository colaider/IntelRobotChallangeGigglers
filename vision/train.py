"""Train an anomaly model (PatchCore by default) on dataset/.

Training uses dataset/training/good only; dataset/test/{good,defective} are for evaluation.

Example:
    python vision/train.py --model patchcore
    python vision/train.py --augment 3          # + 3 realistic variations per good image
    python vision/train.py --roi                # brick-only aligned crops (gripper/table ignored)
"""

import argparse
from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore, Stfpm
from lightning.pytorch import seed_everything

from augment import make_augmented
from roi import align_tree

MODELS = {"patchcore": Patchcore, "padim": Padim, "stfpm": Stfpm}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("dataset"))
    p.add_argument("--name", default="bricks")
    p.add_argument("--normal-dir", default="training/good")
    p.add_argument("--normal-test-dir", default="test/good")
    p.add_argument("--abnormal-dir", default="test/defective")
    p.add_argument("--model", choices=MODELS, default="patchcore")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--accelerator", default="cpu", help="cpu (default; MPS crashes on macOS) | auto | gpu")
    p.add_argument("--results", type=Path, default=Path("results"), help="output root (gitignored)")
    p.add_argument("--augment", type=int, default=0, metavar="N",
                   help="add N realistic variations per good image (rotation/shift/light, no colour change)")
    p.add_argument("--aug-flip", action="store_true", help="allow horizontal flips in --augment")
    p.add_argument("--roi", action="store_true",
                   help="cut out + straighten the brick and grey out gripper/background first (vision/roi.py)")
    p.add_argument("--seed", type=int, default=42, help="fixes PatchCore coreset sampling for fair comparisons")
    p.add_argument("--coreset-ratio", type=float, default=None,
                   help="PatchCore memory-bank fraction (default 0.1). With --augment N use 0.1/(N+1) "
                        "to keep the bank - and so latency and model size - the same as without it")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed, workers=True)
    if args.roi:
        roi_root = (args.results / "roi_dataset").resolve()
        align_tree(args.root, roi_root)
        args.root = roi_root  # train and test on aligned, background-free brick crops
    normal_dir = args.normal_dir
    if args.augment:
        aug_dir = (args.results / "augmented" / args.normal_dir).resolve()
        n = make_augmented(args.root / args.normal_dir, aug_dir, args.augment, args.aug_flip, args.seed)
        print(f"Augmented training set: {n} images in {aug_dir}")
        normal_dir = aug_dir  # absolute path: Folder ignores root for it
    dm = Folder(
        name=args.name,
        root=args.root,
        normal_dir=normal_dir,
        normal_test_dir=args.normal_test_dir,
        abnormal_dir=args.abnormal_dir,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=42,
    )
    kwargs = {}
    if args.coreset_ratio is not None and args.model == "patchcore":
        kwargs["coreset_sampling_ratio"] = args.coreset_ratio
    model = MODELS[args.model](**kwargs)
    engine = Engine(default_root_dir=args.results, accelerator=args.accelerator)
    engine.fit(datamodule=dm, model=model)
    print("\nTest metrics:")
    engine.test(datamodule=dm, model=model)
    print(f"\nCheckpoint: {engine.trainer.checkpoint_callback.best_model_path or 'see ' + str(args.results)}")


if __name__ == "__main__":
    main()
