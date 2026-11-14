"""Post-processing: turn raw sessions into an Anomalib dataset (optionally cropped / CLAHE).

Always starts from the raw files in data/<session>/, so you can try other settings at any time
without collecting again.

    # good from 3 sessions for training, one later session held out for testing, cropped
    py collection/post.py --sessions data/2026*_good* --test-sessions data/20261115_0930_red_test \
        --crop scene --out dataset_robot
    # same, plus CLAHE (adaptive histogram equalisation) - compare both with vision/train.py
    py collection/post.py --sessions data/2026* --test-sessions ... --crop scene --clahe --out dataset_robot_clahe

Output = the layout vision/train.py expects:
    <out>/training/good       good captures from the non-test sessions
    <out>/test/good           good captures from --test-sessions
    <out>/test/defective      every defective capture (any session), file name carries the defect type

Split by SESSION (another sitting/time), never by shuffling one session: photos taken a second
apart are near-duplicates and would make test scores look better than they are.

About CLAHE: it evens out brightness locally and makes small dark marks stand out, but it also
boosts noise and reflections. Do not assume it helps - train with and without and compare.
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vision"))


def clahe(bgr: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """Adaptive histogram equalisation on lightness only (colours stay the same)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def read_manifest(session: Path) -> list[dict]:
    with (session / "manifest.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def prepare(bgr: np.ndarray, crop: str, use_clahe: bool, margin: float) -> np.ndarray | None:
    """crop: none | paper | scene. Returns None if the brick is not found."""
    if crop == "none":
        return clahe(bgr) if use_clahe else bgr
    from segment import FILL, process  # vision/segment.py

    try:
        _, bgr, crop_mask, _ = process(bgr, mode=crop, margin=margin)
    except ValueError:
        return None
    if use_clahe:
        # CLAHE darkens the flat grey background next to the brick (halo); paint it flat again
        keep = cv2.dilate(crop_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        bgr = clahe(bgr)
        bgr[keep == 0] = FILL
    return bgr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sessions", type=Path, nargs="+", required=True, help="session folders (training + test)")
    p.add_argument("--test-sessions", type=Path, nargs="*", default=[],
                   help="sessions whose GOOD captures go to test/good (defective always go to test)")
    p.add_argument("--camera", default=None,
                   help="which camera to export, e.g. ground (default: the first camera of each session). "
                        "One model per camera view: never mix views in one dataset")
    p.add_argument("--crop", choices=["none", "paper", "scene"], default="none")
    p.add_argument("--clahe", action="store_true")
    p.add_argument("--margin", type=float, default=0.08)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sessions = sorted({s.resolve() for s in [*args.sessions, *args.test_sessions] if s.is_dir()})
    test = {s.resolve() for s in args.test_sessions}
    if not test:
        print("WARNING: no --test-sessions: test/good will be empty (Anomalib then has nothing to "
              "threshold on). Collect a separate good session for testing.")
    if args.out.exists():
        shutil.rmtree(args.out)
    for d in ("training/good", "test/good", "test/defective"):
        (args.out / d).mkdir(parents=True)

    counts, failed = {"training/good": 0, "test/good": 0, "test/defective": 0}, []
    for s in sessions:
        cams = list(json.loads((s / "session.json").read_text()).get("cameras", {}))
        cam = args.camera or (cams[0] if cams else "realsense")
        if cam not in cams:
            print(f"skip {s.name}: no camera '{cam}' (has {cams})")
            continue
        for row in read_manifest(s):
            src = s / f"{row['id']}_{cam}_color.png"
            if not src.exists():
                continue
            if row["label"] == "defective":
                split, suffix = "test/defective", f"_{row['defect'] or 'defect'}"
            else:
                split, suffix = ("test/good" if s in test else "training/good"), ""
            img = prepare(cv2.imread(str(src)), args.crop, args.clahe, args.margin)
            if img is None:
                failed.append(f"{s.name}/{src.name}")
                continue
            name = f"{s.name}_{row['id']}{'_' + row['color'] if row['color'] else ''}{suffix}.png"
            cv2.imwrite(str(args.out / split / name), img)
            counts[split] += 1

    print(f"{args.out}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    if failed:
        print(f"brick not found in {len(failed)}: " + ", ".join(failed[:10]) + (" ..." if len(failed) > 10 else ""))
    print(f"train: python vision/train.py --root {args.out}")


if __name__ == "__main__":
    main()
