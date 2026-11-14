"""Find and crop the brick in a GROUND-camera frame (fixed camera, brick held by the gripper).

How it works (no depth: the brick is closer than the D415's minimum range):
  1. background  = per-pixel median of many frames from this camera. The gripper and brick move
                   between frames, the room does not, so the median is the empty scene.
  2. brick pixels = differ from the background AND are not dark (arm, cables) AND are colourful
                   for their brightness (chroma > 0.18 x lightness). The white gripper jaws look
                   bright but almost colourless under the room light, so they drop out.
  3. morphology  = open (specks) -> close (gaps) -> largest blob -> fill holes -> open again
                   (cuts thin jaw fringes) -> convex hull (a brick outline is ~convex, fills the
                   dark inside of an upside-down brick).
  4. no brick    = blob smaller than 2 % of the frame -> None (empty gripper, brick out of view).
  5. crop        = rotate long side horizontal (studs up for side views), tight square crop,
                   background painted flat grey.

If the camera moves or the room changes a lot, rebuild the background (prepare.py does).
Self-contained on purpose (only OpenCV + NumPy), so the robot runtime can import it alone.
"""

from pathlib import Path

import cv2
import numpy as np

FILL = 128  # flat background grey of every crop


def _ellipse(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def largest_blob_filled(mask: np.ndarray) -> np.ndarray:
    """Keep the largest connected component and fill its holes. uint8 0/1 in and out."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    m = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(m)
    cv2.drawContours(out, contours, -1, 1, thickness=cv2.FILLED)
    return out


class BrickFinder:
    def __init__(self, background: np.ndarray, fg_thresh: float = 22, min_lightness: float = 28,
                 min_chroma_ratio: float = 0.18, min_chroma: float = 10, min_area: float = 0.02,
                 margin: float = 0.08):
        self.background = background
        self._bg_lab = self._lab(background)
        self.fg_thresh, self.min_lightness = fg_thresh, min_lightness
        self.min_chroma_ratio, self.min_chroma = min_chroma_ratio, min_chroma
        self.min_area, self.margin = min_area, margin

    # -- background ---------------------------------------------------------
    @staticmethod
    def _lab(bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)

    @classmethod
    def from_frames(cls, frames: list[np.ndarray], **kw) -> "BrickFinder":
        """Median background from frames of the same fixed camera (20+ frames, brick moving)."""
        return cls(np.median(np.stack(frames), axis=0).astype(np.uint8), **kw)

    @classmethod
    def load(cls, path: Path, **kw) -> "BrickFinder":
        bg = cv2.imread(str(path))
        if bg is None:
            raise FileNotFoundError(f"background image not found: {path}")
        return cls(bg, **kw)

    def save(self, path: Path) -> None:
        cv2.imwrite(str(path), self.background)

    # -- finding ------------------------------------------------------------
    def mask(self, bgr: np.ndarray) -> np.ndarray | None:
        """Brick mask (uint8 0/1, frame size) or None when no brick is visible."""
        if bgr.shape != self.background.shape:
            raise ValueError(f"frame {bgr.shape} differs from background {self.background.shape}: "
                             "same camera and resolution as in training?")
        lab = self._lab(bgr)
        fg = np.linalg.norm(lab - self._bg_lab, axis=2) > self.fg_thresh
        light = lab[..., 0]
        chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
        m = (fg & (light > self.min_lightness) & (chroma > self.min_chroma)
             & (chroma > self.min_chroma_ratio * light)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _ellipse(7))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _ellipse(21))
        m = largest_blob_filled(m)
        if m.mean() < self.min_area:
            return None
        m = largest_blob_filled(cv2.morphologyEx(m, cv2.MORPH_OPEN, _ellipse(15)))
        if m.mean() < self.min_area:
            return None
        hull = np.zeros_like(m)
        cv2.fillConvexPoly(hull, cv2.convexHull(cv2.findNonZero(m)), 1)
        return hull

    def crop(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """(square crop with grey background, crop mask) or None when no brick is visible."""
        m = self.mask(bgr)
        return None if m is None else canonical_crop(bgr, m, self.margin)


def _studs_at_bottom(mask_r: np.ndarray, band: float = 0.15) -> bool:
    """Side view rotated long-side-horizontal: the stud edge is bumpier (less filled)."""
    x, y, bw, bh = cv2.boundingRect(mask_r)
    k = max(2, int(band * bh))
    roi = mask_r[y:y + bh, x:x + bw]
    return roi[-k:].mean() < roi[:k].mean()


def canonical_crop(bgr: np.ndarray, mask: np.ndarray, margin: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    """Rotate long side horizontal (studs up), crop tight + margin, grey background, square."""
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(cv2.findNonZero(mask))
    if rw < rh:
        angle -= 90
    h, w = mask.shape
    rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    msk_r = cv2.warpAffine(mask, rot, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    if _studs_at_bottom(msk_r):
        angle += 180
        rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        msk_r = cv2.warpAffine(mask, rot, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    img_r = cv2.warpAffine(bgr, rot, (w, h), flags=cv2.INTER_LINEAR, borderValue=(FILL,) * 3)
    img_r[cv2.dilate(msk_r, _ellipse(7)) == 0] = FILL

    x, y, bw, bh = cv2.boundingRect(msk_r)
    side = max(bw, bh) + 2 * int(round(margin * max(bw, bh)))
    canvas = np.full((side, side, 3), FILL, np.uint8)
    canvas_m = np.zeros((side, side), np.uint8)
    x0, y0 = x + bw // 2 - side // 2, y + bh // 2 - side // 2
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + side, w), min(y0 + side, h)
    canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img_r[sy0:sy1, sx0:sx1]
    canvas_m[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = msk_r[sy0:sy1, sx0:sx1]
    return canvas, canvas_m


def overlay(bgr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Green brick outline on the frame (red text when no brick), for QA pictures."""
    out = bgr.copy()
    if mask is None:
        cv2.putText(out, "NO BRICK", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return out
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    return out
