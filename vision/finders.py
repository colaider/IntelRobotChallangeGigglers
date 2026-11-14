"""Ways to find and crop the brick in a camera frame, behind one interface, for comparisons.

    finder = make_finder("bg_scene", camera="ground", background=bg)
    r = finder(bgr)          # FindResult: crop, crop_mask, frame_mask, ok, reasons, ms

Finders:
    full        no cropping: the whole frame resized to a square (baseline)
    bg          median background of a FIXED camera: pixels that differ from the empty scene,
                are not dark and are colourful for their brightness (idea from Alex's
                anomaly/brick.py). Milliseconds per frame. Needs `background`.
    scene       segment.process(mode="scene"): mean-shift + k-means + GrabCut (~1 s per frame)
    bg_scene    bg mask -> padded box -> scene search only inside that box (scene quality,
                smaller search area)
    window      scene search inside a fixed window (camera-specific, see WINDOWS): for the
                gripper camera the brick always sits between the jaws
    window_raw  just the fixed window, no segmentation at all (cheapest possible)
    window_tight  a tighter fixed window around the brick only (less changing background)

Every finder except `full` / `window_raw` ends in segment.canonical_crop, so crops look the same
(long side horizontal, studs up, grey background) and differ only in how the brick was found.
"""

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from segment import FILL, canonical_crop, clean_mask, mask_quality, process

# fixed windows (x0, y0, x1, y1 as fractions of the frame), from the contact sheets of the
# 2026-11-14 sessions: WINDOWS = search area incl. the jaws, TIGHT = where the brick itself sits
WINDOWS = {"gripper": (0.36, 0.42, 0.70, 0.95)}
TIGHT = {"gripper": (0.41, 0.46, 0.65, 0.84)}


@dataclass
class FindResult:
    crop: np.ndarray | None = None          # square BGR crop (None: no brick found)
    crop_mask: np.ndarray | None = None     # brick mask on the crop (uint8 0/1)
    frame_mask: np.ndarray | None = None    # brick mask on the frame (uint8 0/1)
    ok: bool = False                        # mask quality check passed
    reasons: list[str] = field(default_factory=list)
    ms: float = 0.0


# ---------------------------------------------------------------- background subtraction
def median_background(frames: list[np.ndarray]) -> np.ndarray:
    """Empty scene of a fixed camera = per-pixel median of frames where the brick moves around."""
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def bg_mask(bgr: np.ndarray, background: np.ndarray, fg_thresh: float = 22, min_lightness: float = 28,
            min_chroma: float = 10, min_chroma_ratio: float = 0.18) -> np.ndarray:
    """Brick pixels: differ from the background in Lab, not dark (arm, cables) and colourful for
    their brightness (white gripper jaws are bright but nearly colourless). uint8 0/1."""
    def lab(x):
        return cv2.cvtColor(cv2.GaussianBlur(x, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)

    f, b = lab(bgr), lab(background)
    fg = np.linalg.norm(f - b, axis=2) > fg_thresh
    chroma = np.hypot(f[..., 1] - 128, f[..., 2] - 128)
    m = fg & (f[..., 0] > min_lightness) & (chroma > min_chroma) & (chroma > min_chroma_ratio * f[..., 0])
    m = clean_mask(m.astype(np.uint8))
    if m.any():  # a brick outline is ~convex: the hull fills the dark inside of an upside-down brick
        hull = np.zeros_like(m)
        cv2.fillConvexPoly(hull, cv2.convexHull(cv2.findNonZero(m)), 1)
        m = hull
    return m


def box_window(mask: np.ndarray, pad: float = 0.35) -> tuple[float, float, float, float] | None:
    """Bounding box of the mask, padded by `pad` of its size, as frame fractions."""
    pts = cv2.findNonZero(mask)
    if pts is None:
        return None
    x, y, w, h = cv2.boundingRect(pts)
    H, W = mask.shape
    px, py = int(pad * w), int(pad * h)
    return (max(0, x - px) / W, max(0, y - py) / H, min(W, x + w + px) / W, min(H, y + h + py) / H)


# ---------------------------------------------------------------- finders
def _from_mask(bgr: np.ndarray, mask: np.ndarray, margin: float) -> FindResult:
    if mask is None or mask.mean() < 0.005:
        return FindResult(reasons=["no brick"])
    crop, crop_m, info = canonical_crop(bgr, mask, margin=margin)
    ok, reasons = mask_quality(mask)
    return FindResult(crop, crop_m, mask, ok, reasons)


def _from_process(bgr: np.ndarray, margin: float, window=None) -> FindResult:
    try:
        mask, crop, crop_m, info = process(bgr, mode="scene", window=window, margin=margin)
    except ValueError:
        return FindResult(reasons=["no brick"])
    return FindResult(crop, crop_m, mask, info["ok"], info["reasons"])


def _fixed_window(bgr: np.ndarray, window) -> FindResult:
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = (int(round(v * s)) for v, s in zip(window, (w, h, w, h)))
    crop = bgr[y0:y1, x0:x1]
    side = max(crop.shape[:2])
    canvas = np.full((side, side, 3), FILL, np.uint8)
    oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    cm = np.zeros((side, side), np.uint8)
    cm[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = 1
    fm = np.zeros((h, w), np.uint8)
    fm[y0:y1, x0:x1] = 1
    return FindResult(canvas, cm, fm, True, [])


def make_finder(name: str, camera: str = "ground", background: np.ndarray | None = None,
                margin: float = 0.08):
    """Returns f(bgr) -> FindResult (timed)."""
    window = WINDOWS.get(camera)

    def run(bgr: np.ndarray) -> FindResult:
        if name == "full":
            return _fixed_window(bgr, (0, 0, 1, 1))
        if name == "window_raw":
            return _fixed_window(bgr, window)
        if name == "window_tight":
            return _fixed_window(bgr, TIGHT[camera])
        if name == "scene":
            return _from_process(bgr, margin)
        if name == "window":
            return _from_process(bgr, margin, window)
        if name in ("bg", "bg_scene"):
            if background is None:
                raise ValueError(f"finder {name} needs a background image")
            m = bg_mask(bgr, background)
            if name == "bg":
                return _from_mask(bgr, m, margin)
            win = box_window(m)
            return FindResult(reasons=["no brick"]) if win is None else _from_process(bgr, margin, win)
        raise ValueError(f"unknown finder {name}")

    def timed(bgr: np.ndarray) -> FindResult:
        t = time.perf_counter()
        r = run(bgr)
        r.ms = (time.perf_counter() - t) * 1000
        return r

    return timed


FINDERS = ["full", "bg", "scene", "bg_scene", "window", "window_raw", "window_tight"]
NEEDS_BACKGROUND = {"bg", "bg_scene"}
NEEDS_WINDOW = {"window", "window_raw", "window_tight"}
