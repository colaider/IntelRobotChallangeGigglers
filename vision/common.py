"""Shared helpers for the vision scripts: model registry, datamodule, test-set listing, metrics."""

from pathlib import Path

import numpy as np
from anomalib.data import Folder
from anomalib.models import Dinomaly, EfficientAd, Padim, Patchcore, Stfpm

# name -> (class, max_epochs). One-epoch models just build a memory bank / statistics.
MODELS = {
    "patchcore": (Patchcore, 1),
    "padim": (Padim, 1),
    "stfpm": (Stfpm, 30),
    "efficient_ad": (EfficientAd, 30),  # downloads ImageNette (~1.5 GB) on first run
    "dinomaly": (Dinomaly, 30),  # DINOv2 backbone, heavy
}

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def make_datamodule(root: Path, normal_dir="training/good", normal_test_dir="test/good",
                    abnormal_dir="test/defective", batch_size=8, num_workers=0,
                    resize: int | None = None) -> Folder:
    """resize: square size applied to every split (needed for INT8 calibration of static models)."""
    extra = {}
    if resize:
        from torchvision.transforms.v2 import Resize
        extra["augmentations"] = Resize((resize, resize), antialias=True)
    return Folder(
        **extra,
        name="bricks",
        root=root,
        normal_dir=normal_dir,
        normal_test_dir=normal_test_dir,
        abnormal_dir=abnormal_dir,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=num_workers,
        seed=42,
    )


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXT)


def list_test_set(root: Path, good="test/good", defective="test/defective") -> list[tuple[Path, int]]:
    """(image, label) pairs; label 1 = defective."""
    return [(p, 0) for p in list_images(root / good)] + [(p, 1) for p in list_images(root / defective)]


def auroc(labels, scores) -> float:
    """Rank-based AUROC (probability a defective image scores higher than a good one)."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (len(pos) * len(neg)))


def f1(labels, preds) -> float:
    labels, preds = np.asarray(labels).astype(bool), np.asarray(preds).astype(bool)
    tp = (labels & preds).sum()
    fp = (~labels & preds).sum()
    fn = (labels & ~preds).sum()
    return float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0
