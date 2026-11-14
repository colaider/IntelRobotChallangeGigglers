"""Turn robot collection sessions (data/<session>/ + dataset/faces.csv) into Anomalib datasets per
camera x face, with the brick already cut out. Re-run it on every new collection.

Cameras are never mixed (gripper and front see the brick completely differently). Per camera we
build one dataset per face AND one with all faces, so the model variants can be compared:

    <camera>_<face>   gripper_top, gripper_bottom, gripper_side, front_top, front_bottom, front_side
                      -> variant "per_face" (one model per face; face picked by competition or classifier)
    <camera>_all      gripper_all, front_all
                      -> variant "single" (one model for every face of that camera)

Face labels (dataset/faces.csv): `face` = what the front camera ("ground") sees,
`gripper_face` = what the gripper camera sees (empty = top). `none` / `occluded` frames are skipped.
Brick cut-out per camera: gripper = fixed window between the jaws + segmentation, front =
difference to the median empty scene (vision/finders.py) - the same finders as at runtime.

Output (gitignored), ready for train_views.py (or train.py / export.py / calibrate.py, no --seg):
    dataset/views/<view>/training/good/*.png
    dataset/views/<view>/test/good/*.png          whole arm-pose groups held out, per colour
    dataset/views/<view>/test/defective/*.png     frames of sessions labelled defective
    dataset/views/<view>/qa.jpg, skipped.txt      LOOK at qa.jpg before training

Example:
    python vision/build_views.py                          # every camera x face + *_all
    python vision/build_views.py --cameras front --test-frac 0.3
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2

from finders import make_finder, median_background
from segment import sheet
from sessions import load_captures

CAMERAS = {  # view prefix -> (camera name in data/, finder, which label field)
    "gripper": ("gripper", "window", "gripper_face"),
    "front": ("ground", "bg", "face"),
}
FACES = ["top", "bottom", "side"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--faces", type=Path, default=Path("dataset/faces.csv"))
    p.add_argument("--out", type=Path, default=Path("dataset/views"))
    p.add_argument("--cameras", nargs="+", choices=CAMERAS, default=list(CAMERAS))
    p.add_argument("--test-frac", type=float, default=0.25, help="share of arm-pose groups held out per colour")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def face_of(c, field: str) -> str:
    f = (getattr(c, field) or "").strip()
    return f if f else ("top" if field == "gripper_face" else "")


def split(caps, test_frac: float, seed: int) -> dict[str, str]:
    """key -> train/test; whole pose groups move together (near-duplicates never leak), per colour."""
    rng = random.Random(seed)
    strata = defaultdict(lambda: defaultdict(list))
    for c in caps:
        strata[c.colour][c.pose_group].append(c)
    out = {}
    for colour in sorted(strata):
        groups = sorted(strata[colour])
        rng.shuffle(groups)
        n_test = max(1, round(test_frac * len(groups))) if len(groups) > 2 else 0
        for i, g in enumerate(groups):
            for c in strata[colour][g]:
                out[c.key] = "test" if i < n_test else "train"
    return out


def main() -> None:
    args = parse_args()
    caps_all = load_captures(args.data, args.faces)
    if not caps_all:
        raise SystemExit(f"no sessions in {args.data}")
    summary = []
    for prefix in args.cameras:
        camera, finder_name, field = CAMERAS[prefix]
        caps = [c for c in caps_all if face_of(c, field) in FACES]
        if not caps:
            print(f"{prefix}: no labelled frames (fill `{field}` in {args.faces}), skipped")
            continue
        background = median_background([c.image(camera) for c in caps_all]) if finder_name == "bg" else None
        if background is not None:  # the runtime finder needs the same empty-scene reference
            args.out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.out / f"{prefix}_background.png"), background)
        find = make_finder(finder_name, camera, background)
        good = [c for c in caps if c.label != "defective"]
        sp = split(good, args.test_frac, args.seed)  # one split per camera: same frames held out in every view

        views = [f"{prefix}_{f}" for f in FACES] + [f"{prefix}_all"]
        for v in views:
            shutil.rmtree(args.out / v, ignore_errors=True)
            for sub in ("training/good", "test/good", "test/defective"):
                (args.out / v / sub).mkdir(parents=True, exist_ok=True)
        tiles, skipped = defaultdict(list), defaultdict(list)
        counts = defaultdict(lambda: defaultdict(int))
        for c in caps:
            face = face_of(c, field)
            sub = "test/defective" if c.label == "defective" else (
                "training/good" if sp[c.key] == "train" else "test/good")
            r = find(c.image(camera))
            targets = [f"{prefix}_{face}", f"{prefix}_all"]
            if r.crop is None or not r.ok:
                for v in targets:
                    skipped[v].append(f"{c.key} ({sub}): {', '.join(r.reasons) or 'no brick'}")
                continue
            for v in targets:
                cv2.imwrite(str(args.out / v / sub / f"{c.key}.png"), r.crop)
                counts[v][sub] += 1
                t = r.crop.copy()
                cv2.putText(t, f"{sub.split('/')[0][:5]} {face}", (6, 24), 0, 0.6, (0, 255, 255), 2)
                tiles[v].append(t)
        for v in views:
            if tiles[v]:
                cv2.imwrite(str(args.out / v / "qa.jpg"), sheet(tiles[v], cols=10, size=140))
            (args.out / v / "skipped.txt").write_text("\n".join(skipped[v]) + "\n")
            k = counts[v]
            summary.append(f"{v:15s} train {k['training/good']:3d}  test good {k['test/good']:3d}  "
                           f"test defective {k['test/defective']:3d}  skipped {len(skipped[v]):3d}")
    print("\n".join(summary))
    print(f"\nLook at {args.out}/<view>/qa.jpg. Views without test/defective frames train on Anomalib's "
          "synthetic anomalies (indicative only). Next: python vision/train_views.py")


if __name__ == "__main__":
    main()
