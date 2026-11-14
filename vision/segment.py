"""Find the brick, straighten it and crop it tight: the pre-processing step before Anomalib.

Why: in the raw 700x700 captures the brick covers only ~7-18 % of the frame, its size changes
from photo to photo, and PatchCore spends most of its memory bank on paper, shadows and
highlights. After this step every image is "one brick, long side horizontal, filling the
frame, flat grey background", so small defects are bigger at 256x256 and the background can
no longer cause false alarms.

Pipeline (same function for capture, training and robot inference):
    raw mask  ->  morphology (open, close, largest blob, fill holes)  ->  minAreaRect
              ->  rotate so the long side is horizontal  ->  tight crop + margin
              ->  background painted flat grey  ->  pad to square

Raw mask sources:
    brick_mask_rgb    colour distance from the paper in Lab (works for pale lilac too, unlike
                      HSV saturation); used for the photos we already have
    brick_mask_depth  height above the table from an aligned RealSense depth frame (UNTESTED
                      until we have the camera; depth ignores colour, shadows and highlights)

Example (re-process an existing dataset, writes images + masks + a QA overlay sheet):
    python vision/segment.py --root dataset --out dataset_seg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp")
SPLITS = ["training/good", "test/good", "test/defective"]
FILL = 128  # flat background grey


# ---------------------------------------------------------------- raw masks
def brick_mask_rgb(bgr: np.ndarray, min_dist: float = 8.0) -> np.ndarray:
    """Pixels whose colour (Lab a*, b*) differs from the paper. Returns uint8 0/1.

    The paper colour is the median of a border strip, so a slightly blue or warm sheet is fine.
    Lightness L is ignored on purpose: shadows are darker paper, not brick.
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    b = max(4, min(h, w) // 25)
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    a0, b0 = np.median(border[:, 1]), np.median(border[:, 2])
    dist = np.hypot(lab[..., 1] - a0, lab[..., 2] - b0)
    # Otsu on the distance image picks the paper/brick split per photo; never go below min_dist
    d8 = np.clip(dist * 4, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (dist > max(min_dist, otsu / 4)).astype(np.uint8)


def brick_mask_scene(bgr: np.ndarray, k: int = 8, work: int = 320, refine_px: int = 960) -> np.ndarray:
    """Brick mask on a CLUTTERED scene (brick in the gripper, table, cables, hands, lights).

    No paper to compare against, so: flatten texture (mean-shift), split into k colour regions
    (k-means in Lab), and pick the connected region that looks most like a brick:
        big (sqrt area) x convex (solidity^2) x central (Gaussian) / textured / touches border
    then refine its outline with GrabCut. Works for white bricks too (no colour needed).
    Returns uint8 0/1 at full resolution.
    """
    h, w = bgr.shape[:2]
    s = work / max(h, w)
    small = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    flat = cv2.pyrMeanShiftFiltering(small, 10, 18)
    data = cv2.cvtColor(flat, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(0)
    _, labels, _ = cv2.kmeans(data, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                              3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(small.shape[:2])
    grey = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)
    texture = np.abs(cv2.Laplacian(grey, cv2.CV_32F))
    sh, sw = labels.shape
    best, best_score = None, 0.0
    for c in range(k):
        m = cv2.morphologyEx((labels == c).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, cc, st, cen = cv2.connectedComponentsWithStats(m)
        for i in range(1, n):
            area = st[i, cv2.CC_STAT_AREA] / m.size
            if not 0.03 <= area <= 0.6:
                continue
            comp = (cc == i).astype(np.uint8)
            x, y, bw, bh = st[i, :4]
            border = int(x == 0) + int(y == 0) + int(x + bw >= sw) + int(y + bh >= sh)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            solidity = st[i, cv2.CC_STAT_AREA] / max(cv2.contourArea(cv2.convexHull(cnts[0])), 1)
            dist = np.hypot((cen[i][0] - sw / 2) / sw, (cen[i][1] - sh / 2) / sh)
            score = (np.sqrt(area) * solidity ** 2 * np.exp(-4 * dist ** 2)
                     / (1 + texture[comp > 0].mean() / 4) / (1 + border))
            if score > best_score:
                best, best_score = comp, score
    if best is None:
        return np.zeros((h, w), np.uint8)

    # GrabCut at a medium resolution: seed = region (probable fg), its core (sure fg),
    # a ring around it (probable bg), everything else sure background
    r_s = min(1.0, refine_px / max(h, w))  # never upscale (640x480 stays as is)
    mid = cv2.resize(bgr, (int(w * r_s), int(h * r_s)), interpolation=cv2.INTER_AREA)
    mh, mw = mid.shape[:2]
    comp = cv2.resize(best, (mw, mh), interpolation=cv2.INTER_NEAREST)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(3, int(0.04 * min(mh, mw))) + 1,) * 2)
    gc = np.full((mh, mw), cv2.GC_BGD, np.uint8)
    gc[cv2.dilate(comp, ker, iterations=2) > 0] = cv2.GC_PR_BGD
    gc[comp > 0] = cv2.GC_PR_FGD
    gc[cv2.erode(comp, ker) > 0] = cv2.GC_FGD
    cv2.grabCut(mid, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_MASK)
    fg = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    # strong opening cuts thin bits of hand / gripper that GrabCut glued to the brick
    k_cut = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, min(mh, mw) // 35) | 1,) * 2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k_cut)
    return cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)


def brick_mask_depth(depth_mm: np.ndarray, table_mm: np.ndarray | float,
                     min_height_mm: float = 5.0, max_height_mm: float = 80.0) -> np.ndarray:
    """Height-above-table mask from a RealSense depth frame aligned to colour (UNTESTED).

    depth_mm: depth in mm (0 = no data). table_mm: depth of the empty table (a saved frame or a
    single number for a flat, square-on table). A Duplo brick is ~19 mm tall, so 5 mm is safe.
    """
    height = np.asarray(table_mm, np.float32) - depth_mm.astype(np.float32)
    return ((depth_mm > 0) & (height > min_height_mm) & (height < max_height_mm)).astype(np.uint8)


# ---------------------------------------------------------------- morphology
def clean_mask(raw: np.ndarray) -> np.ndarray:
    """Morphological clean-up shared by both mask sources. Returns uint8 0/1 (may be empty).

    open    removes speckle / depth flying pixels
    close   bridges small gaps (studs vs body, thin dark edges)
    largest connected component drops clutter (other objects, gripper parts)
    fill holes - a dark defect breaks the colour/depth mask from the inside; without filling,
            we would cut out exactly the defect we are looking for
    """
    s = min(raw.shape[:2])
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, s // 140) | 1,) * 2)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, s // 45) | 1,) * 2)
    m = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k_open)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(m)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == biggest).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled


# ---------------------------------------------------------------- crop
def studs_at_bottom(mask_r: np.ndarray, band: float = 0.15) -> bool:
    """On a brick already rotated long-side-horizontal: is the stud side at the bottom?

    The studs make the outline bumpy, so the edge band on the stud side is less filled than
    the flat side. Compares mask fill in the top and bottom `band` of the bounding box.
    """
    x, y, bw, bh = cv2.boundingRect(mask_r)
    k = max(2, int(band * bh))
    roi = mask_r[y:y + bh, x:x + bw]
    return roi[-k:].mean() < roi[:k].mean()


def canonical_crop(bgr: np.ndarray, mask: np.ndarray, margin: float = 0.08, fill: int = FILL,
                   keep_px: int = 3) -> tuple[np.ndarray, np.ndarray, dict]:
    """Rotate so the brick's long side is horizontal, crop tight, grey background, square pad.

    keep_px: the mask is dilated by this much before painting the background, so the brick's
    outer edge (where edge defects live) is never painted over.
    Returns (crop, crop_mask, info) with info = angle, rect size, area fraction.
    """
    pts = cv2.findNonZero(mask)
    if pts is None:
        raise ValueError("empty mask: no brick found")
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(pts)
    if rw < rh:  # make the long side horizontal
        angle -= 90
        rw, rh = rh, rw
    h, w = mask.shape
    rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    msk_r = cv2.warpAffine(mask, rot, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    if studs_at_bottom(msk_r):  # minAreaRect cannot tell up from down; put the studs on top
        angle += 180
        rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        msk_r = cv2.warpAffine(mask, rot, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    img_r = cv2.warpAffine(bgr, rot, (w, h), flags=cv2.INTER_LINEAR, borderValue=(fill,) * 3)

    keep = cv2.dilate(msk_r, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * keep_px + 1,) * 2))
    img_r[keep == 0] = fill

    x, y, bw, bh = cv2.boundingRect(msk_r)
    pad = int(round(margin * max(bw, bh)))
    side = max(bw, bh) + 2 * pad
    # square canvas centred on the brick; areas outside the photo stay grey
    canvas = np.full((side, side, 3), fill, np.uint8)
    canvas_m = np.zeros((side, side), np.uint8)
    x0, y0 = x + bw // 2 - side // 2, y + bh // 2 - side // 2
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + side, w), min(y0 + side, h)
    canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img_r[sy0:sy1, sx0:sx1]
    canvas_m[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = msk_r[sy0:sy1, sx0:sx1]
    info = {"angle": float(angle), "rect": (float(rw), float(rh)), "area_frac": float(mask.mean())}
    return canvas, canvas_m, info


MODES = {"paper": brick_mask_rgb, "scene": brick_mask_scene}


def raw_mask(bgr: np.ndarray, mode: str = "paper", depth_mm=None, table_mm=None) -> np.ndarray:
    """Depth if given; else 'paper' (plain background) or 'scene' (cluttered, brick in gripper)."""
    if depth_mm is not None:
        return brick_mask_depth(depth_mm, table_mm)
    return MODES[mode](bgr)


def process(bgr: np.ndarray, mode: str = "paper", depth_mm=None, table_mm=None, **kw):
    """One call for everything: raw mask -> clean -> crop. Returns (mask, crop, crop_mask, info)."""
    mask = clean_mask(raw_mask(bgr, mode, depth_mm, table_mm))
    return (mask, *canonical_crop(bgr, mask, **kw))


# ---------------------------------------------------------------- CLI
def overlay(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 3)
    pts = cv2.findNonZero(mask)
    if pts is not None:
        box = cv2.boxPoints(cv2.minAreaRect(pts)).astype(np.int32)
        cv2.polylines(out, [box], True, (255, 0, 255), 2)
    return out


def sheet(tiles: list[np.ndarray], cols: int = 9, size: int = 160) -> np.ndarray:
    tiles = [cv2.resize(t, (size, size), interpolation=cv2.INTER_AREA) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros((size, size, 3), np.uint8))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("dataset"))
    p.add_argument("--out", type=Path, default=Path("dataset_seg"), help="output root (gitignored)")
    p.add_argument("--splits", nargs="+", default=SPLITS)
    p.add_argument("--margin", type=float, default=0.08, help="crop margin as a fraction of the brick size")
    p.add_argument("--image", type=Path, default=None,
                   help="process ONE photo: writes <out>/<name>_crop.png and <name>_steps.png (every stage)")
    p.add_argument("--mode", choices=MODES, default="paper",
                   help="paper = brick on a plain sheet; scene = cluttered background (brick in the gripper)")
    return p.parse_args()


def label(img: np.ndarray, text: str, size: int = 300) -> np.ndarray:
    t = cv2.resize(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), (size, size))
    cv2.putText(t, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(t, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return t


def one_image(path: Path, out: Path, margin: float, mode: str = "paper") -> None:
    """Run the pipeline on one photo and save a strip with every stage, for learning/debugging."""
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"cannot read {path}")
    raw = raw_mask(img, mode)
    mask = clean_mask(raw)
    crop, crop_m, info = canonical_crop(img, mask, margin=margin)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"{path.stem}_crop.png"), crop)
    strip = np.hstack([
        label(img, "1 photo"),
        label(raw * 255, f"2 raw mask ({mode})"),
        label(mask * 255, "3 after morphology"),
        label(overlay(img, mask), "4 outline + min rect"),
        label(crop, f"5 crop (rot {info['angle']:.0f} deg)"),
    ])
    cv2.imwrite(str(out / f"{path.stem}_steps.png"), strip)
    print(f"brick {100 * info['area_frac']:.1f}% of frame, rotated {info['angle']:.1f} deg, "
          f"crop {crop.shape[1]}x{crop.shape[0]} px")
    print(f"-> {out / (path.stem + '_crop.png')}  (feed this to predict.py)")
    print(f"-> {out / (path.stem + '_steps.png')}  (what each stage did)")


def main() -> None:
    args = parse_args()
    if args.image:
        one_image(args.image, args.out, args.margin, args.mode)
        return
    qa_over, qa_crop, failed = [], [], []
    for split in args.splits:
        src = args.root / split
        files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXT)
        dst, dst_m = args.out / split, args.out / "masks" / split
        dst.mkdir(parents=True, exist_ok=True)
        dst_m.mkdir(parents=True, exist_ok=True)
        for f in files:
            img = cv2.imread(str(f))
            try:
                mask, crop, crop_m, info = process(img, mode=args.mode, margin=args.margin)
            except ValueError as e:
                failed.append(f"{split}/{f.name}: {e}")
                continue
            cv2.imwrite(str(dst / f.name), crop)
            cv2.imwrite(str(dst_m / f.name), crop_m * 255)
            qa_over.append(overlay(img, mask))
            qa_crop.append(crop)
            print(f"{split:16s} {f.name:32s} brick {100 * info['area_frac']:5.1f}% of frame  "
                  f"rot {info['angle']:6.1f} deg  crop {crop.shape[1]}px")
    if qa_over:
        cv2.imwrite(str(args.out / "qa_masks.jpg"), sheet(qa_over))
        cv2.imwrite(str(args.out / "qa_crops.jpg"), sheet(qa_crop))
        print(f"\nQA sheets: {args.out / 'qa_masks.jpg'}, {args.out / 'qa_crops.jpg'}")
    if failed:
        print("FAILED:\n  " + "\n  ".join(failed))


if __name__ == "__main__":
    main()
