"""Brick ROI: find the brick in a frame, cut it out, straighten it, grey out everything else.

Why: the anomaly model should only judge the brick. Gripper, table, cables, shadows next to the
brick change from frame to frame and would be "anomalies". The mask is computed per frame from
the brick itself, so it works for any brick position or gripper pose (no fixed pixel mask).

Steps:
  1. background colour = median of the frame border
  2. brick pixels = chroma (Lab a,b) far from the background + saturation (Otsu thresholds)
  3. largest blob, holes filled (marker strokes inside the brick stay in the mask!), slightly dilated
  4. rotate so the brick's long side is horizontal, crop with padding, grey out non-brick pixels,
     letterbox to a square of --size px

Example:
    python vision/roi.py --input dataset --out dataset_roi            # whole dataset tree
    python vision/train.py --root dataset_roi
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp")
GREY = 128


def brick_mask(bgr: np.ndarray, dilate_frac: float = 0.01) -> np.ndarray | None:
    """uint8 0/1 mask of the brick (filled), or None if nothing brick-like is found."""
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    border = np.concatenate([lab[:5].reshape(-1, 3), lab[-5:].reshape(-1, 3),
                             lab[:, :5].reshape(-1, 3), lab[:, -5:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    # chroma distance ignores lightness, so grey shadows on the table do not count as brick
    chroma = np.linalg.norm(lab[..., 1:] - bg[1:], axis=-1)
    chroma = cv2.GaussianBlur(np.clip(chroma * 4, 0, 255).astype(np.uint8), (5, 5), 0)
    sat = cv2.GaussianBlur(hsv[..., 1], (5, 5), 0)
    _, m1 = cv2.threshold(chroma, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, m2 = cv2.threshold(sat, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (m1 | m2).astype(np.uint8)
    k = max(3, int(min(h, w) * 0.01)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3 * k, 3 * k), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.005 * h * w:
        return None
    filled = np.zeros((h, w), np.uint8)
    cv2.drawContours(filled, [cv2.convexHull(c)], -1, 1, -1)  # hull: marker on the edge stays inside
    d = max(1, int(min(h, w) * dilate_frac)) * 2 + 1
    return cv2.dilate(filled, np.ones((d, d), np.uint8))


def align(bgr: np.ndarray, size: int = 256, pad: float = 0.08, mask_outside: bool = True):
    """Return (aligned square crop, mask) or (None, None) if no brick was found."""
    mask = brick_mask(bgr)
    if mask is None:
        return None, None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    if rw < rh:  # make the long side horizontal
        angle += 90
        rw, rh = rh, rw
    img = bgr.copy()
    if mask_outside:
        img[mask == 0] = GREY
    # rotate around the brick centre on a canvas large enough to keep the whole brick
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rot = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=(GREY,) * 3)
    rmask = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    side = int(max(rw, rh) * (1 + 2 * pad))
    x0, y0 = int(cx - side / 2), int(cy - side / 2)
    canvas = np.full((side, side, 3), GREY, np.uint8)
    mcanvas = np.zeros((side, side), np.uint8)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + side, w), min(y0 + side, h)
    canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rot[sy0:sy1, sx0:sx1]
    mcanvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rmask[sy0:sy1, sx0:sx1]
    out = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    return out, cv2.resize(mcanvas, (size, size), interpolation=cv2.INTER_NEAREST)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="image folder or dataset root (walked recursively)")
    p.add_argument("--out", type=Path, required=True, help="mirror of --input with aligned crops")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--pad", type=float, default=0.08, help="margin around the brick, fraction of its length")
    p.add_argument("--keep-background", action="store_true", help="crop/straighten but do not grey out")
    return p.parse_args()


def align_tree(src: Path, dst: Path, size: int = 256, pad: float = 0.08, mask_outside: bool = True):
    """Align every image under src (file or folder) into the same relative path under dst.

    Returns the list of files where no brick was found (they are skipped).
    """
    files = [src] if src.is_file() else sorted(
        p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXT and "masks" not in p.parts)
    failed = []
    for p in files:
        out, _ = align(cv2.imread(str(p)), size, pad, mask_outside)
        if out is None:
            failed.append(p)
            continue
        target = dst / (p.name if src.is_file() else p.relative_to(src))
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), out)
    print(f"ROI: {len(files) - len(failed)}/{len(files)} aligned into {dst}")
    for p in failed:
        print(f"  no brick found (skipped): {p}")
    return failed


def main() -> None:
    args = parse_args()
    align_tree(args.input, args.out, args.size, args.pad, not args.keep_background)


if __name__ == "__main__":
    main()
