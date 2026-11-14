"""Brick finders for the LeRobot recordings: 'where the brick must be' + colour (single frame).

side_colour_bg  side camera (main, must be right every time): brick = coloured AND different from a
                background reference (median of frames without the brick in place: table, bowls,
                backdrop are static). Then shadow removal, convex hull filled only with coloured /
                glare pixels, a small grow into the darker bevel. Settings: 3 px border (keep_px=3),
                no opening - the variant picked on the 49 recordings (49/49 complete crops).
gripper_roi_clean  gripper camera (extra, quick screen): the camera rides with the jaws, so the brick
                is the coloured blob in a fixed window between them. Pixels that move while the brick
                is carried (table / hall) are dropped, then only pixels near the brick's own hue are
                kept (wood is saturated too), an erosion+dilation removes glued-on bits, 1 px border.
                43/49 clean, the rest rejected by mask_quality (too small / not convex).
Both end in segment.clean_mask + canonical_crop + mask_quality, so crops match the rest of the code.
"""
import cv2
import numpy as np

from segment import canonical_crop, clean_mask, mask_quality, refine_by_colour

BLUE = (85, 135)                     # OpenCV hue range of the blue bowl (never the brick)
GRIP_ROI = (0.36, 0.46, 0.62, 0.82)  # window between the jaws, fractions of the gripper frame
SIDE = dict(keep_px=3, open_frac=0.0)
GRIPPER = dict(keep_px=1, open_frac=0.15, hue_trim=8, glare=(40, 190))


def colour_mask(bgr, min_s=40, min_v=55):
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    otsu, _ = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(min_s, min(otsu, 90))
    m = (s > thr) & (v > min_v) & ~((h > BLUE[0]) & (h < BLUE[1]))
    return m.astype(np.uint8)


def _blob(mask, prefer_xy=None, drop_border=False, bridge=15, min_frac=0.01):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    # glare lines / stud edges split a brick into pieces: bridge small gaps before picking a blob
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge, bridge)))
    n, cc, st, cen = cv2.connectedComponentsWithStats(m)
    h, w = m.shape
    best, best_score = None, 0
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if area < min_frac * h * w:
            continue
        touches = (x == 0) + (y == 0) + (x + bw >= w) + (y + bh >= h)
        if drop_border and touches >= 1:
            # the brick may touch one edge when it is close to the frame border; table / bowls span it
            if bw > 0.6 * w or touches >= 2:
                continue
        score = area
        if prefer_xy is not None:
            d = np.hypot((cen[i][0] - prefer_xy[0]) / w, (cen[i][1] - prefer_xy[1]) / h)
            score = area * np.exp(-6 * d * d)
        if score > best_score:
            best, best_score = (cc == i).astype(np.uint8), score
    return best


def _finish(bgr, mask, hull=True, fg=None, refine=None, keep_px=None, open_frac=None, hue_trim=None, glare=(60, 0),
            min_area=0.02):
    keep_px = 3 if keep_px is None else keep_px
    open_frac = open_frac or 0.0
    if mask is None or mask.sum() == 0:
        raise ValueError("no brick found")
    mask = clean_mask(mask)
    if refine if refine is not None else fg is not None:  # drop off-colour pieces (shadow, wood, jaws)
        mask = refine_by_colour(bgr, mask)
    if hull:
        # glare on glossy plastic has no colour and leaves notches; a brick face is convex. Fill the
        # hull only with coloured or bright (glare) pixels, never with dark gripper / arm parts.
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h = np.zeros_like(mask)
        cv2.drawContours(h, [cv2.convexHull(np.vstack(cnts))], -1, 1, -1)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        allowed = (colour_mask(bgr, min_s=25) > 0) | ((hsv[..., 2] > 170) & (hsv[..., 1] < 80))
        allowed = cv2.dilate(allowed.astype(np.uint8), np.ones((7, 7), np.uint8))
        if fg is not None:  # never fill with static background (table, paper)
            allowed &= fg
        mask = clean_mask((h & allowed).astype(np.uint8))
        # keep the brick's outer edge (darker bevel, defects live there): grow the mask a little,
        # only into pixels that still look like the brick (coloured or a shaded version of it)
        r = max(2, int(0.015 * np.sqrt(mask.sum())))
        grown = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref = np.median(lab[mask > 0], axis=0)
        like = (np.hypot(lab[..., 1] - ref[1], lab[..., 2] - ref[2]) < 25) & (lab[..., 0] > 0.45 * ref[0])
        if fg is not None:
            like &= fg > 0
        mask = clean_mask((mask | (grown & like.astype(np.uint8))).astype(np.uint8))
    if hue_trim:
        # gripper: wood / skin around the brick is also saturated; keep only pixels whose hue is near
        # the brick's own hue (taken from the mask core), unsaturated glare stays
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        core = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        core = core if core.sum() > 200 else mask
        hh = hsv[..., 0].astype(np.int16)
        ang = hh[core > 0].astype(np.float32) * np.pi / 90
        ref = int(round(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) * 90 / np.pi)) % 180
        dh = np.abs(hh - ref); dh = np.minimum(dh, 180 - dh)
        keep = (dh <= hue_trim) | ((hsv[..., 1] < glare[0]) & (hsv[..., 2] > glare[1]))
        m2 = (mask & keep).astype(np.uint8)
        m2 = cv2.morphologyEx(m2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        if m2.any():
            n, cc, st, _ = cv2.connectedComponentsWithStats(m2)
            mask = clean_mask((cc == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8))
    if open_frac > 0:
        # erosion then dilation: removes bits narrower than the kernel (wood / jaw pieces glued to
        # the brick) and gives the brick its size back
        k = max(3, int(open_frac * np.sqrt(mask.sum()))) | 1
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        if opened.any():
            n, cc, st, _ = cv2.connectedComponentsWithStats(opened)
            mask = clean_mask((cc == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8))
    crop, crop_m, info = canonical_crop(bgr, mask, keep_px=keep_px)
    info["ok"], info["reasons"] = mask_quality(mask, min_area=min_area)
    return mask, crop, crop_m, info


def gripper_roi(bgr, roi=GRIP_ROI):
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)
    sub = colour_mask(bgr[y0:y1, x0:x1], min_s=35)
    blob = _blob(sub, prefer_xy=((x1 - x0) / 2, (y1 - y0) / 2), bridge=1)
    full = np.zeros((h, w), np.uint8)
    if blob is not None:
        full[y0:y1, x0:x1] = blob
    mask, crop, crop_m, info = _finish(bgr, full, hull=False, refine=False, keep_px=1)
    # the ROI border is not the frame border: drop that reason
    info["reasons"] = [r for r in info["reasons"] if "border" not in r]
    info["ok"] = not info["reasons"]
    return mask, crop, crop_m, info


def side_colour_bg(bgr, background, diff_thr=25):
    """Side camera with a background reference (median of the episode's frames): table, bowls and
    backdrop are static -> only the moving arm + brick differ; the brick is the coloured part."""
    lab = lambda x: cv2.cvtColor(cv2.GaussianBlur(x, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    d = np.linalg.norm(lab(bgr) - lab(background), axis=-1)
    fg = (d > diff_thr).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    m = colour_mask(bgr, min_s=30) & fg
    fg_wide = cv2.dilate(fg, np.ones((5, 5), np.uint8))
    return _finish(bgr, _blob(m), fg=fg_wide, **SIDE)


def side_small(bgr, background, diff_thr=25):
    """Side camera, brick SMALL in the frame (side faces: the arm holds the brick below the jaws,
    ~70x55 px = ~0.5 % of a 640x480 frame). Same idea as side_colour_bg (coloured AND different from
    the background) with size limits for a small brick. Pale bricks (lilac) are not found: almost the
    colour of the white backdrop and jaws - left out of the side dataset on purpose."""
    lab = lambda x: cv2.cvtColor(cv2.GaussianBlur(x, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    d = np.linalg.norm(lab(bgr) - lab(background), axis=-1)
    fg = (d > diff_thr).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    m = colour_mask(bgr, min_s=30) & fg
    fg_wide = cv2.dilate(fg, np.ones((3, 3), np.uint8))
    return _finish(bgr, _blob(m, bridge=7, min_frac=0.002), fg=fg_wide, min_area=0.002, **SIDE)


def gripper_roi_clean(bgr, carry_frames, roi=GRIP_ROI, move_thr=1.0, **kw):
    """gripper_roi, then drop only clearly moving pixels (table / hall seen around the brick while it
    was carried): temporal spread above move_thr x the frame's median spread. Holes are filled."""
    mask, _, _, _ = gripper_roi(bgr, roi)
    Ls = []
    for f in carry_frames:
        L = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        Ls.append(L / max(L.mean(), 1.0) * 128.0)
    sd = np.stack(Ls).std(0)
    moving = (sd > move_thr * float(np.median(sd))).astype(np.uint8)
    moving = cv2.morphologyEx(moving, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))  # only big moving areas
    m = (mask & (1 - moving)).astype(np.uint8)
    m = _blob(m, bridge=5) if m.any() else m
    out = _finish(bgr, m, hull=False, refine=False, **(GRIPPER | kw))
    out[3]["reasons"] = [r for r in out[3]["reasons"] if "border" not in r]
    out[3]["ok"] = not out[3]["reasons"]
    return out
