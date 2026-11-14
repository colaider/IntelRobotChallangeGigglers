"""Sensitivity table: which defect sizes / marker colours does a model catch, per brick colour?

Scores a sweep made by synth_defects.py --sweep (files <stem>_sweep_d<px>_<ink>.png) and the
clean images in <root>/test/good with an exported OpenVINO model, using the model's own
threshold (score >= 0.5 = defective).

Example:
    python vision/synth_defects.py --input dataset/test/good --out dataset_sweep --sweep 4 8 12 16 24 32
    python vision/sensitivity.py --xml results/export/patchcore/fp16/weights/openvino/model.xml \
        --root dataset_sweep
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from benchmark import Scorer
from synth_defects import brick_colour, list_images

NAME_RE = re.compile(r"^(?P<stem>.+)_sweep_d(?P<d>\d+)_(?P<ink>[a-z]+)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xml", type=Path, required=True, help="exported OpenVINO model.xml")
    p.add_argument("--root", type=Path, default=Path("dataset_sweep"))
    p.add_argument("--device", default="CPU")
    p.add_argument("--px-per-mm", type=float, default=10.0)
    p.add_argument("--by-colour", action="store_true", help="one table per brick colour")
    return p.parse_args()


def score(scorer: Scorer, path: Path) -> tuple[float, bool, np.ndarray]:
    bgr = cv2.imread(str(path))
    r = scorer(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return float(np.asarray(r.pred_score).ravel()[0]), bool(np.asarray(r.pred_label).ravel()[0]), bgr


def print_table(title: str, hits: dict, px_per_mm: float) -> None:
    sizes = sorted({d for d, _ in hits})
    inks = sorted({i for _, i in hits})
    print(f"\n{title}  (detected / total)")
    print(f"{'diameter':>14s} " + " ".join(f"{i:>8s}" for i in inks) + f" {'all':>8s}")
    for d in sizes:
        cells, tot_hit, tot_n = [], 0, 0
        for ink in inks:
            h, n = hits.get((d, ink), (0, 0))
            tot_hit, tot_n = tot_hit + h, tot_n + n
            cells.append(f"{h:>3d}/{n:<4d}" if n else f"{'-':>8s}")
        print(f"{d:>4d}px {d / px_per_mm:4.1f}mm " + " ".join(cells) + f" {100 * tot_hit / max(tot_n, 1):7.0f}%")


def main() -> None:
    args = parse_args()
    scorer = Scorer(args.xml, args.device)

    colour_of = {}  # stem -> brick colour, from the clean image
    flagged = set()  # stems whose CLEAN image is already called defective
    good = list_images(args.root / "test/good")
    false_alarms = defaultdict(lambda: [0, 0])
    for p in good:
        s, label, bgr = score(scorer, p)
        colour_of[p.stem] = brick_colour(bgr)
        if label:
            flagged.add(p.stem)
        fa = false_alarms[colour_of[p.stem]]
        fa[0] += label
        fa[1] += 1

    # A dot on a brick that is already a false alarm says nothing about sensitivity:
    # only count defects on bricks whose clean version passes as good.
    hits = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # colour -> (d, ink) -> [hit, n]
    skipped = 0
    for p in list_images(args.root / "test/defective"):
        m = NAME_RE.match(p.stem)
        if not m:
            continue
        if m["stem"] in flagged:
            skipped += 1
            continue
        _, label, _ = score(scorer, p)
        colour = colour_of.get(m["stem"], "?") if args.by_colour else "all bricks"
        cell = hits[colour][(int(m["d"]), m["ink"])]
        cell[0] += label
        cell[1] += 1

    for colour, table in sorted(hits.items()):
        print_table(f"[{colour}]", {k: tuple(v) for k, v in table.items()}, args.px_per_mm)

    print("\nFalse alarms on clean test/good:")
    for colour, (fa, n) in sorted(false_alarms.items()):
        print(f"  {colour:8s} {fa}/{n}")
    if skipped:
        print(f"({skipped} defect images skipped: their clean brick is already a false alarm)")


if __name__ == "__main__":
    main()
