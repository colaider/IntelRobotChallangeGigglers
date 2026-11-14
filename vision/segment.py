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
    r_s = refine_px / max(h, w)
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


def brick_mask_scene_window(bgr: np.ndarray, window=None, **kw) -> np.ndarray:
    """brick_mask_scene inside a search window (x0, y0, x1, y1 as fractions of the frame).

    On the station the arm always presents the brick at the same pose, so we know roughly where
    it is before looking: searching only there removes table / people / balloons from the
    candidate regions and makes the search faster.
    """
    if window is None:
        return brick_mask_scene(bgr, **kw)
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = (int(round(v * s)) for v, s in zip(window, (w, h, w, h)))
    out = np.zeros((h, w), np.uint8)
    out[y0:y1, x0:x1] = brick_mask_scene(bgr[y0:y1, x0:x1], **kw)
    return out


def refine_by_colour(bgr: np.ndarray, mask: np.ndarray, max_de: float = 22.0) -> np.ndarray:
    """Remove pieces of hand / gripper / background that segmentation glued to the brick.

    1. reference colour = median Lab of the mask core (eroded), i.e. the brick itself
    2. keep mask pixels close to that colour (lightness counts half: shading/highlights are
       the same plastic), take the largest blob
    3. convex hull of that blob AND the original mask, then fill holes. The hull restores
       marker strokes on the brick edge (dark, so dropped in step 2) but not bits that stick
       out of the brick outline.
    """
    if mask.sum() == 0:
        return mask
    s = int(np.sqrt(mask.sum()))
    core = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, s // 6) | 1,) * 2))
    if core.sum() < 50:
        core = mask
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = np.median(lab[core > 0], axis=0)
    de = np.sqrt((0.5 * (lab[..., 0] - ref[0])) ** 2 + (lab[..., 1] - ref[1]) ** 2 + (lab[..., 2] - ref[2]) ** 2)
    keep = ((de < max_de) & (mask > 0)).astype(np.uint8)
    keep = cv2.morphologyEx(keep, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, cc, st, _ = cv2.connectedComponentsWithStats(keep)
    if n <= 1:
        return mask
    blob = (cc == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = np.zeros_like(mask)
    cv2.drawContours(hull, [cv2.convexHull(np.vstack(cnts))], -1, 1, -1)
    return clean_mask(hull & mask)


def mask_quality(mask: np.ndarray, min_area: float = 0.02, max_area: float = 0.6,
                 aspect: tuple[float, float] = (1.0, 3.2), min_solidity: float = 0.80) -> tuple[bool, list[str]]:
    """Is this a plausible single brick? Returns (ok, reasons). Not ok -> treat as 'uncertain'.

    A Duplo brick seen from any side is a solid box: rect aspect 1:1 (2x2 top) .. ~3:1 (2x4 side
    with studs), high solidity (studs make it slightly bumpy), not tiny, not the whole frame,
    not cut by the frame border.
    """
    pts = cv2.findNonZero(mask)
    if pts is None:
        return False, ["empty mask"]
    reasons = []
    area = float(mask.mean())
    if not min_area <= area <= max_area:
        reasons.append(f"area {100 * area:.1f}% of frame")
    (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
    ar = max(rw, rh) / max(min(rw, rh), 1)
    if not aspect[0] <= ar <= aspect[1]:
        reasons.append(f"aspect {ar:.2f}")
    solidity = float(mask.sum()) / max(cv2.contourArea(cv2.convexHull(pts)), 1)
    if solidity < min_solidity:
        reasons.append(f"solidity {solidity:.2f}")
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        reasons.append("touches frame border")
    return not reasons, reasons


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
    # orientation from the brick body only: a strong opening removes studs and glued-on bits,
    # which would otherwise tilt the rectangle
    k = max(3, int(0.12 * np.sqrt(mask.sum()))) | 1
    body = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    body_pts = cv2.findNonZero(body)
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(body_pts if body_pts is not None else pts)
    if rw < rh:  # make the long side horizontal
        angle -= 90
        rw, rh = rh, rw
    h, w = mask.shape

    def rotated(a):
        r = cv2.getRotationMatrix2D((cx, cy), a, 1.0)
        m = trim_outline(cv2.warpAffine(mask, r, (w, h), flags=cv2.INTER_NEAREST, borderValue=0))
        im = cv2.warpAffine(bgr, r, (w, h), flags=cv2.INTER_LINEAR, borderValue=(fill,) * 3)
        return (r, *box_constrain(m, rh, img_r=im))

    rot, msk_r, studs_down = rotated(angle)
    if studs_down:  # minAreaRect cannot tell up from down; put the studs on top
        angle += 180
        rot, msk_r, _ = rotated(angle)
    img_r = cv2.warpAffine(bgr, rot, (w, h), flags=cv2.INTER_LINEAR, borderValue=(fill,) * 3)
    msk_r = snap_sides(img_r, msk_r)
    trimmed = 1 - msk_r.sum() / max(mask.sum(), 1)

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
    info = {"angle": float(angle), "rect": (float(rw), float(rh)), "area_frac": float(mask.mean()),
            "trimmed": float(trimmed)}
    return canvas, canvas_m, info


def snap_sides(img_r: np.ndarray, mask_r: np.ndarray, search: float = 0.12, min_ratio: float = 1.8) -> np.ndarray:
    """Cut a pale gripper / hand strip glued to the left or right side of a straightened brick.

    Colour cannot always tell a pale brick from a pale-grey gripper, but the brick's side is a
    straight vertical edge. For each side, look inside the outer `search` fraction of the mask
    width for the column with the strongest horizontal colour gradient (Lab, summed over the
    body rows); if it is clearly stronger than typical (min_ratio x median), cut beyond it.
    """
    pts = cv2.findNonZero(mask_r)
    if pts is None:
        return mask_r
    x, y, bw, bh = cv2.boundingRect(pts)
    lab = cv2.cvtColor(cv2.GaussianBlur(img_r, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    gx = sum(np.abs(cv2.Sobel(lab[..., c], cv2.CV_32F, 1, 0, ksize=3)) for c in range(3))
    rows = slice(y + int(0.25 * bh), y + bh - int(0.1 * bh))  # body rows (studs are on top)
    prof = gx[rows, x:x + bw].sum(0)
    if prof.size < 10:
        return mask_r
    med = float(np.median(prof)) + 1e-6
    n = max(3, int(search * bw))

    def inner_edge(p):
        """Index of the strongest edge INSIDE the outer mask boundary (skip the boundary peak)."""
        i = int(np.argmax(p[:max(4, n // 3)]))          # outer boundary peak
        while i + 1 < n and p[i + 1] <= p[i]:           # walk down to the valley after it
            i += 1
        if i + 2 >= n:
            return None
        j = i + 1 + int(np.argmax(p[i + 1:n]))
        return j if p[j] > min_ratio * med and p[j] > 1.3 * p[i] else None

    out = mask_r.copy()
    li = inner_edge(prof)
    if li is not None:
        out[:, :x + li] = 0
    ri = inner_edge(prof[::-1])
    if ri is not None:
        out[:, x + bw - ri:] = 0
    return out


def box_constrain(mask_r: np.ndarray, short_side: float, stud_frac: float = 0.45,
                  slack: float = 0.015, img_r: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
    """Brick geometry on a mask rotated long-side-horizontal. Returns (mask, studs_at_bottom).

    1. body = the box without protrusions: an opening wider than a stud removes studs and thin
       glued-on bits (hand, gripper)
    2. studs side = where the mask sticks out of the body, counted only over the central body
       columns (bits at the corners or sides do not vote)
    3. keep only what fits a brick: body + small slack on three sides, + up to stud height on
       the studs side. Everything else (gripper / hand bits) is cut.
    """
    if mask_r.sum() == 0:
        return mask_r, False
    k = max(5, int(stud_frac * short_side)) | 1
    body = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    pts = cv2.findNonZero(body)
    if pts is None:
        return mask_r, studs_at_bottom(mask_r)
    bx, by, bw, bh = cv2.boundingRect(pts)
    c0, c1 = bx + int(0.1 * bw), bx + bw - int(0.1 * bw)
    above = int(mask_r[:by, c0:c1].sum())
    below = int(mask_r[by + bh:, c0:c1].sum())
    # no visible protrusion either way (studs merged into the body by the mask, or a face
    # without studs): fall back to the edge-fill heuristic
    studs_down = below > above if above != below else studs_at_bottom(mask_r)
    sx, sy = int(slack * bw), int(slack * bh)
    stud = int(stud_frac * bh) + sy
    y0, y1 = (by - sy, by + bh + stud) if studs_down else (by - stud, by + bh + sy)
    box = np.zeros_like(mask_r)
    box[max(y0, 0):max(y1, 0), max(bx - sx, 0):bx + bw + sx] = 1
    out = (mask_r & box).astype(np.uint8)
    # stud band: keep only stud-sized blobs; thin slivers of hand / gripper beside the studs go
    band = np.zeros_like(out)
    if studs_down:
        band[by + bh + sy:] = out[by + bh + sy:]
    else:
        band[:max(by - sy, 0)] = out[:max(by - sy, 0)]
    band_all = band.copy()  # everything in the stud band; only what passes the filters comes back
    if band.any() and img_r is not None:
        # studs have the brick's hue; a greenish gripper / pink skin sliver does not. Chroma only
        # (a*, b*), so studs in shadow are kept.
        lab = cv2.cvtColor(cv2.GaussianBlur(img_r, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
        bodypx = lab[by:by + bh, bx:bx + bw][mask_r[by:by + bh, bx:bx + bw] > 0]
        if len(bodypx) > 50:
            ref = np.median(bodypx, axis=0)
            chroma = np.hypot(lab[..., 1] - ref[1], lab[..., 2] - ref[2])
            spread = float(np.median(np.hypot(bodypx[:, 1] - ref[1], bodypx[:, 2] - ref[2])))
            band = (band & (chroma < max(6.0, 4 * spread))).astype(np.uint8)
    if band.any():
        ks = max(3, int(0.12 * bh)) | 1
        studs = cv2.morphologyEx(band, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
        n, cc, st, _ = cv2.connectedComponentsWithStats(studs)
        keep = np.zeros_like(band)
        if n > 1:
            big = st[1:, cv2.CC_STAT_AREA].max()
            for i in range(1, n):
                if st[i, cv2.CC_STAT_AREA] >= 0.25 * big:
                    keep[cc == i] = 1
        # stud blobs survive the opening slightly shrunk: take the original band pixels near them
        keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))) & band
        out = ((out & (band_all == 0)) | keep).astype(np.uint8)
    elif band_all.any():
        out = (out & (band_all == 0)).astype(np.uint8)
    return out, studs_down


def _main_run(on: np.ndarray) -> np.ndarray:
    """Boolean vector -> only its longest contiguous True run stays True."""
    best, start, best_rng = 0, None, (0, 0)
    for i, v in enumerate(np.append(on, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best:
                best, best_rng = i - start, (start, i)
            start = None
    out = np.zeros_like(on)
    out[best_rng[0]:best_rng[1]] = True
    return out


def trim_outline(mask_r: np.ndarray, col_frac: float = 0.5, row_frac: float = 0.2) -> np.ndarray:
    """On a brick rotated long-side-horizontal, cut thin bits sticking out of its box.

    Brick body columns are filled over (almost) the whole height; a glued-on bit of hand or
    gripper at the side fills only a few pixels of a column -> drop columns below col_frac of
    the fullest one. Rows use a low threshold so the stud row (two studs ~ 40 % of the width)
    survives. Only the main contiguous band is kept; marker strokes inside the brick are not
    affected (the mask is filled).
    """
    if mask_r.sum() == 0:
        return mask_r
    cols, rows = mask_r.sum(0), mask_r.sum(1)
    keep_c = _main_run(cols >= col_frac * cols.max())
    keep_r = _main_run(rows >= row_frac * rows.max())
    return (mask_r * keep_c[None, :] * keep_r[:, None]).astype(np.uint8)


MODES = {"paper": brick_mask_rgb, "scene": brick_mask_scene}


def raw_mask(bgr: np.ndarray, mode: str = "paper", depth_mm=None, table_mm=None, window=None) -> np.ndarray:
    """Depth if given; else 'paper' (plain background) or 'scene' (cluttered, brick in gripper)."""
    if depth_mm is not None:
        return brick_mask_depth(depth_mm, table_mm)
    if mode == "scene":
        return brick_mask_scene_window(bgr, window)
    return MODES[mode](bgr)


def process(bgr: np.ndarray, mode: str = "paper", depth_mm=None, table_mm=None, window=None,
            refine: bool = True, **kw):
    """One call for everything: raw mask -> clean -> colour refine -> crop.

    Returns (mask, crop, crop_mask, info). info["ok"] / info["reasons"] = mask quality check:
    robot code treats not ok as 'uncertain' (re-grasp / re-inspect instead of guessing).
    """
    mask = clean_mask(raw_mask(bgr, mode, depth_mm, table_mm, window))
    if refine and depth_mm is None:
        mask = refine_by_colour(bgr, mask)
    crop, crop_m, info = canonical_crop(bgr, mask, **kw)
    info["ok"], info["reasons"] = mask_quality(mask)
    if info["trimmed"] > 0.08:  # a lot was cut off: segmentation grabbed something else too
        info["ok"] = False
        info["reasons"].append(f"trimmed {100 * info['trimmed']:.0f}% of the mask")
    return mask, crop, crop_m, info


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
    p.add_argument("--root", type=Path, default=Path("dataset/raw"))
    p.add_argument("--out", type=Path, default=Path("dataset_seg"), help="output root (gitignored)")
    p.add_argument("--window", type=float, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"),
                   help="scene mode: search only here (fractions of the frame), e.g. 0.25 0.1 0.75 0.9")
    p.add_argument("--no-refine", action="store_true", help="skip the colour clean-up of glued-on bits")
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


def one_image(path: Path, out: Path, margin: float, mode: str = "paper", window=None) -> None:
    """Run the pipeline on one photo and save a strip with every stage, for learning/debugging."""
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"cannot read {path}")
    raw = raw_mask(img, mode, window=window)
    cleaned = clean_mask(raw)
    mask = refine_by_colour(img, cleaned)
    crop, crop_m, info = canonical_crop(img, mask, margin=margin)
    ok, reasons = mask_quality(mask)
    if info["trimmed"] > 0.08:
        ok = False
        reasons.append(f"trimmed {100 * info['trimmed']:.0f}% of the mask")
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"{path.stem}_crop.png"), crop)
    strip = np.hstack([
        label(img, "1 photo"),
        label(raw * 255, f"2 raw mask ({mode})"),
        label(cleaned * 255, "3 after morphology"),
        label(mask * 255, "4 colour refine"),
        label(overlay(img, mask), "5 outline + min rect"),
        label(crop, f"6 crop ({'ok' if ok else 'CHECK'})"),
    ])
    if not ok:
        print("quality check failed:", ", ".join(reasons))
    cv2.imwrite(str(out / f"{path.stem}_steps.png"), strip)
    print(f"brick {100 * info['area_frac']:.1f}% of frame, rotated {info['angle']:.1f} deg, "
          f"crop {crop.shape[1]}x{crop.shape[0]} px")
    print(f"-> {out / (path.stem + '_crop.png')}  (feed this to predict.py)")
    print(f"-> {out / (path.stem + '_steps.png')}  (what each stage did)")


def main() -> None:
    args = parse_args()
    if args.image:
        one_image(args.image, args.out, args.margin, args.mode, args.window)
        return
    segment_tree(args.root, args.out, args.mode, args.splits, args.margin, args.window, not args.no_refine)


def segment_tree(root: Path, out: Path, mode: str = "scene", splits=SPLITS, margin: float = 0.08,
                 window=None, refine: bool = True, verbose: bool = True) -> list[str]:
    """Segment every image of root/<split> into out/<split> (+ masks, QA sheets).

    Images that fail the quality check are still written (so the dataset stays complete) but
    listed; returns that list. Images where no brick is found are skipped.
    """
    qa_over, qa_crop, failed, doubtful = [], [], [], []
    for split in splits:
        src = root / split
        files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXT) if src.exists() else []
        dst, dst_m = out / split, out / "masks" / split
        dst.mkdir(parents=True, exist_ok=True)
        dst_m.mkdir(parents=True, exist_ok=True)
        for f in files:
            img = cv2.imread(str(f))
            try:
                mask, crop, crop_m, info = process(img, mode=mode, margin=margin, window=window, refine=refine)
            except ValueError as e:
                failed.append(f"{split}/{f.name}: {e}")
                continue
            cv2.imwrite(str(dst / f.name), crop)
            cv2.imwrite(str(dst_m / f.name), crop_m * 255)
            qa_over.append(overlay(img, mask))
            qa_crop.append(crop)
            if not info["ok"]:
                doubtful.append(f"{split}/{f.name}: {', '.join(info['reasons'])}")
            if verbose:
                print(f"{split:16s} {f.name:32s} brick {100 * info['area_frac']:5.1f}% of frame  "
                      f"rot {info['angle']:6.1f} deg  crop {crop.shape[1]}px  {'ok' if info['ok'] else 'CHECK'}")
    if qa_over:
        cv2.imwrite(str(out / "qa_masks.jpg"), sheet(qa_over))
        cv2.imwrite(str(out / "qa_crops.jpg"), sheet(qa_crop))
        print(f"QA sheets: {out / 'qa_masks.jpg'}, {out / 'qa_crops.jpg'}  <- look at them")
    if doubtful:
        print("QUALITY CHECK (would be 'uncertain' on the robot):\n  " + "\n  ".join(doubtful))
    if failed:
        print("NO BRICK FOUND (skipped):\n  " + "\n  ".join(failed))
    return doubtful


if __name__ == "__main__":
    main()
