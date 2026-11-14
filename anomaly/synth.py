"""Draw synthetic marker defects on good brick crops (test only, never for training).

Kinds: line, curve, dot, tiny (1-2 mm), scribble, edge (near the brick outline).
Ink is semi-opaque and slightly soft, like a felt-tip marker, and stays inside the brick mask.
"""

import random

import cv2
import numpy as np

KINDS = ["line", "curve", "dot", "tiny", "scribble", "edge"]
INKS = {"black": (25, 25, 25), "blue": (140, 40, 20)}  # BGR


def _point(mask: np.ndarray, rng: random.Random, prefer: np.ndarray | None = None) -> tuple[int, int]:
    ys, xs = np.nonzero(prefer if prefer is not None and prefer.any() else mask)
    i = rng.randrange(len(xs))
    return int(xs[i]), int(ys[i])


def draw(img: np.ndarray, mask: np.ndarray, kind: str, rng: random.Random,
         ink: tuple[int, int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (image with the mark, mark mask uint8 0/255)."""
    h, w = mask.shape
    s = min(h, w)
    ink = ink or INKS[rng.choice(list(INKS))]
    stroke = np.zeros((h, w), np.uint8)
    thick = max(2, int(s * rng.uniform(0.006, 0.012)))
    inner = cv2.erode(mask, np.ones((int(s * .06) | 1,) * 2, np.uint8))
    if kind in ("line", "edge"):
        x0, y0 = _point(mask, rng, (mask - inner) if kind == "edge" else inner)
        ang, length = rng.uniform(0, np.pi), s * rng.uniform(0.08, 0.25)
        cv2.line(stroke, (x0, y0), (int(x0 + length * np.cos(ang)), int(y0 + length * np.sin(ang))), 255, thick,
                 cv2.LINE_AA)
    elif kind == "curve":
        x, y = _point(mask, rng, inner)
        pts, ang = [(x, y)], rng.uniform(0, 2 * np.pi)
        for _ in range(12):
            ang += rng.uniform(-0.5, 0.5)
            x, y = x + s * 0.015 * np.cos(ang), y + s * 0.015 * np.sin(ang)
            pts.append((x, y))
        cv2.polylines(stroke, [np.int32(pts)], False, 255, thick, cv2.LINE_AA)
    elif kind == "scribble":
        x, y = _point(mask, rng, inner)
        pts = [(x + rng.gauss(0, s * .025), y + rng.gauss(0, s * .025)) for _ in range(8)]
        cv2.polylines(stroke, [np.int32(pts)], False, 255, max(2, thick - 1), cv2.LINE_AA)
    elif kind == "dot":
        cv2.circle(stroke, _point(mask, rng, inner), int(s * rng.uniform(0.01, 0.02)), 255, -1, cv2.LINE_AA)
    elif kind == "tiny":
        cv2.circle(stroke, _point(mask, rng, inner), max(2, int(s * rng.uniform(0.003, 0.007))), 255, -1,
                   cv2.LINE_AA)
    else:
        raise ValueError(f"unknown kind {kind}")
    stroke = stroke * (mask > 0)
    alpha = cv2.GaussianBlur(stroke.astype(np.float32) / 255, (3, 3), 0)[..., None] * rng.uniform(0.75, 0.92)
    out = img.astype(np.float32) * (1 - alpha) + np.array(ink, np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8), (stroke > 127).astype(np.uint8) * 255
