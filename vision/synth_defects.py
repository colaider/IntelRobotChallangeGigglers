"""Draw synthetic marker defects on good brick photos.

Use it to get a large defective test set (and pixel masks) without drawing on real bricks,
e.g. to tune thresholds or see from which size a dot is still detected.
Never mix synthetic images into training: they are for evaluation only.

The brick is found as the saturated (coloured) region on a white/grey background; defects are
placed inside it. Kinds: line, curve, dot, tiny (1-2 mm dot), scribble, edge (near brick edge).

Example:
    python vision/synth_defects.py --input dataset/test/good --out dataset_synth --per-image 3
    # -> dataset_synth/test/defective/*.png, dataset_synth/test/masks/*.png,
    #    dataset_synth/test/good (copies) and training/good (symlink-free copies) for train.py
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXT)


KINDS = ["line", "curve", "dot", "tiny", "scribble", "edge"]
INKS = {"black": (25, 25, 25), "blue": (140, 40, 20), "red": (30, 30, 170), "green": (40, 120, 30)}  # BGR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="folder with good images")
    p.add_argument("--out", type=Path, default=Path("dataset_synth"), help="output dataset root (gitignored)")
    p.add_argument("--per-image", type=int, default=3, help="defective variants per good image")
    p.add_argument("--kinds", nargs="+", choices=KINDS, default=KINDS)
    p.add_argument("--inks", nargs="+", choices=list(INKS), default=["black", "blue"])
    p.add_argument("--sat-thresh", type=int, default=60, help="HSV saturation threshold for the brick mask")
    p.add_argument("--copy-good", type=Path, default=None,
                   help="also copy this training/good folder so dataset_synth is a full dataset for train.py")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def brick_mask(img: np.ndarray, sat_thresh: int) -> np.ndarray:
    """Brick mask from segment.py (Lab colour distance + morphology; also finds pale lilac).

    Falls back to HSV saturation, then to the central 60%, if that finds nothing.
    """
    from segment import brick_mask_rgb, clean_mask
    mask = clean_mask(brick_mask_rgb(img))
    if mask.sum() >= 0.02 * mask.size:
        return mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = (hsv[..., 1] > sat_thresh).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (lab == biggest).astype(np.uint8)
    if mask.sum() < 0.02 * mask.size:
        h, w = mask.shape
        mask = np.zeros_like(mask)
        mask[int(.2 * h):int(.8 * h), int(.2 * w):int(.8 * w)] = 1
    return mask


def random_point(mask: np.ndarray, rng: random.Random, inner: np.ndarray | None = None) -> tuple[int, int]:
    ys, xs = np.nonzero(inner if inner is not None and inner.any() else mask)
    i = rng.randrange(len(xs))
    return int(xs[i]), int(ys[i])


def draw_defect(img, mask, kind, ink, rng: random.Random):
    """Return (image, defect_mask)."""
    h, w = mask.shape
    s = min(h, w)  # scale everything to image size
    stroke = np.zeros((h, w), np.uint8)
    thick = max(2, int(s * rng.uniform(0.006, 0.012)))
    inner = cv2.erode(mask, np.ones((int(s * .06) | 1,) * 2, np.uint8))

    if kind in ("line", "edge"):
        if kind == "edge":
            edge = mask - inner
            x0, y0 = random_point(mask, rng, edge)
        else:
            x0, y0 = random_point(mask, rng, inner)
        ang, length = rng.uniform(0, np.pi), s * rng.uniform(0.08, 0.25)
        x1, y1 = int(x0 + length * np.cos(ang)), int(y0 + length * np.sin(ang))
        cv2.line(stroke, (x0, y0), (x1, y1), 255, thick, cv2.LINE_AA)
    elif kind == "curve":
        x0, y0 = random_point(mask, rng, inner)
        pts = [(x0, y0)]
        ang = rng.uniform(0, 2 * np.pi)
        for _ in range(12):
            ang += rng.uniform(-0.5, 0.5)
            x0, y0 = x0 + s * 0.015 * np.cos(ang), y0 + s * 0.015 * np.sin(ang)
            pts.append((x0, y0))
        cv2.polylines(stroke, [np.int32(pts)], False, 255, thick, cv2.LINE_AA)
    elif kind == "scribble":
        x0, y0 = random_point(mask, rng, inner)
        pts = [(x0 + rng.gauss(0, s * .025), y0 + rng.gauss(0, s * .025)) for _ in range(8)]
        cv2.polylines(stroke, [np.int32(pts)], False, 255, max(2, thick - 1), cv2.LINE_AA)
    elif kind == "dot":
        cv2.circle(stroke, random_point(mask, rng, inner), int(s * rng.uniform(0.01, 0.02)), 255, -1, cv2.LINE_AA)
    elif kind == "tiny":
        cv2.circle(stroke, random_point(mask, rng, inner), max(2, int(s * rng.uniform(0.003, 0.007))), 255, -1,
                   cv2.LINE_AA)

    stroke = stroke * mask  # ink stays on the brick
    # marker ink: semi-opaque, slightly soft, keeps a bit of the surface texture/highlights
    alpha = cv2.GaussianBlur(stroke.astype(np.float32) / 255, (3, 3), 0)[..., None] * rng.uniform(0.75, 0.92)
    out = img.astype(np.float32) * (1 - alpha) + np.array(ink, np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8), (stroke > 127).astype(np.uint8) * 255


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    good = list_images(args.input)
    if not good:
        raise SystemExit(f"No images in {args.input}")

    defect_dir, mask_dir = args.out / "test/defective", args.out / "test/masks"
    good_dir = args.out / "test/good"
    for d in (defect_dir, mask_dir, good_dir):
        d.mkdir(parents=True, exist_ok=True)

    for p in good:
        shutil.copy2(p, good_dir / p.name)
        img = cv2.imread(str(p))
        mask = brick_mask(img, args.sat_thresh)
        for k in range(args.per_image):
            kind, ink_name = rng.choice(args.kinds), rng.choice(args.inks)
            out, dmask = draw_defect(img, mask, kind, INKS[ink_name], rng)
            name = f"{p.stem}_syn{k}_{kind}_{ink_name}.png"
            cv2.imwrite(str(defect_dir / name), out)
            cv2.imwrite(str(mask_dir / name), dmask)

    if args.copy_good:
        dst = args.out / "training/good"
        dst.mkdir(parents=True, exist_ok=True)
        for p in list_images(args.copy_good):
            shutil.copy2(p, dst / p.name)

    print(f"{len(good)} good -> {len(good) * args.per_image} synthetic defective in {defect_dir}")
    print(f"masks in {mask_dir}; test/good copied to {good_dir}")


if __name__ == "__main__":
    main()
