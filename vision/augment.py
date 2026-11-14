"""Make extra *good* training images by small, realistic variations of the real ones.

PatchCore/PaDiM learn "normal" from every patch they see, so variations widen what counts as
normal. Only variations the real pipeline produces are allowed:
  - small rotation / shift / scale  (brick lies a bit differently each time)
  - brightness / contrast           (lamp or room light changes a little)
  - optional horizontal flip        (only for faces that are mirror-symmetric)
Deliberately NOT done: hue / saturation changes - that would make "another colour" normal and
hide coloured marker defects.

Example:
    python vision/augment.py --input dataset/training/good --out results/aug/training/good --copies 3
    (train.py --augment 3 does this automatically)
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--copies", type=int, default=3, help="augmented copies per image (originals are kept)")
    p.add_argument("--flip", action="store_true", help="allow horizontal flips (symmetric faces only)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def vary(img: np.ndarray, rng: random.Random, flip: bool, rot=8.0, shift=0.03, scale=0.05,
         bright=0.10, contrast=0.10) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-rot, rot), 1 + rng.uniform(-scale, scale))
    m[:, 2] += (rng.uniform(-shift, shift) * w, rng.uniform(-shift, shift) * h)
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    if flip and rng.random() < 0.5:
        out = cv2.flip(out, 1)
    a = 1 + rng.uniform(-contrast, contrast)
    b = 255 * rng.uniform(-bright, bright)
    mean = out.mean()
    return np.clip((out.astype(np.float32) - mean) * a + mean + b, 0, 255).astype(np.uint8)


def make_augmented(src: Path, dst: Path, copies: int, flip: bool = False, seed: int = 0) -> int:
    """Copy originals + `copies` variants of each into dst (cleared first). Returns image count."""
    rng = random.Random(seed)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    n = 0
    for p in sorted(q for q in src.rglob("*") if q.suffix.lower() in IMAGE_EXT):
        shutil.copy2(p, dst / p.name)
        img = cv2.imread(str(p))
        for k in range(copies):
            cv2.imwrite(str(dst / f"{p.stem}_aug{k}.png"), vary(img, rng, flip))
        n += 1 + copies
    return n


def main() -> None:
    args = parse_args()
    n = make_augmented(args.input, args.out, args.copies, args.flip, args.seed)
    print(f"{n} images in {args.out}")


if __name__ == "__main__":
    main()
