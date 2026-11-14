"""Turn robot collection sessions (data/<session>/ + dataset/faces.csv) into one Anomalib dataset per
inspection VIEW, with the brick already cut out. TEMPORARY bridge until the clean booth data arrives;
re-run it on every new collection.

Views (team plan, docs/inspection_plan_2cam.md):
    gripper_top   gripper camera after the grasp from above -> sees the top face   (finder: window)
    front_side    front camera (D415 "ground"), a side face square-on              (finder: bg)
    front_bottom  front camera, the bottom / cavity                                 (finder: bg)
    front_top     front camera, top face (only if such frames exist)                (finder: bg)

Output (gitignored), ready for train.py / export.py / calibrate.py (no --seg: crops are brick-only):
    dataset/views/<view>/training/good/*.png
    dataset/views/<view>/test/good/*.png          whole arm-pose groups held out, per colour
    dataset/views/<view>/test/defective/*.png     frames of sessions labelled defective (if any)
    dataset/views/<view>/qa.jpg, skipped.txt      look at qa.jpg before training

Then per view:
    python vision/train.py --root dataset/views/front_side
    python vision/export.py --ckpt <ckpt> --root dataset/views/front_side
    python vision/calibrate.py --xml <model.xml> --seg none \
        --good dataset/views/front_side/test/good --defective dataset/views/front_side/test/defective

Example:
    python vision/build_views.py                   # all views, data/ -> dataset/views/
    python vision/build_views.py --views front_side --test-frac 0.3
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

VIEWS = {  # view -> (camera, ground-camera face label or None = any brick in the gripper, finder)
    "gripper_top": ("gripper", None, "window"),
    "front_side": ("ground", "side", "bg"),
    "front_bottom": ("ground", "bottom", "bg"),
    "front_top": ("ground", "top", "bg"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--faces", type=Path, default=Path("dataset/faces.csv"))
    p.add_argument("--out", type=Path, default=Path("dataset/views"))
    p.add_argument("--views", nargs="+", choices=VIEWS, default=list(VIEWS))
    p.add_argument("--test-frac", type=float, default=0.25, help="share of arm-pose groups held out per colour")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


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
    backgrounds = {}
    for view in args.views:
        camera, face, finder_name = VIEWS[view]
        if camera == "gripper":
            caps = [c for c in caps_all if c.face != "occluded"]  # brick in the gripper, any ground view
        else:
            caps = [c for c in caps_all if c.face == face]
        good = [c for c in caps if c.label != "defective"]
        bad = [c for c in caps if c.label == "defective"]
        if not good:
            print(f"{view}: no frames, skipped")
            continue
        if finder_name == "bg" and camera not in backgrounds:  # fixed camera: empty scene = median frame
            backgrounds[camera] = median_background([c.image(camera) for c in caps_all])
        find = make_finder(finder_name, camera, backgrounds.get(camera))

        root = args.out / view
        shutil.rmtree(root, ignore_errors=True)
        sp = split(good, args.test_frac, args.seed)
        jobs = [(c, "training/good" if sp[c.key] == "train" else "test/good") for c in good]
        jobs += [(c, "test/defective") for c in bad]
        tiles, skipped, counts = [], [], defaultdict(int)
        for c, sub in jobs:
            r = find(c.image(camera))
            if r.crop is None or not r.ok:
                skipped.append(f"{c.key} ({sub}): {', '.join(r.reasons) or 'no brick'}")
                continue
            d = root / sub
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{c.key}.png"), r.crop)
            counts[sub] += 1
            t = r.crop.copy()
            cv2.putText(t, sub.split("/")[0][:5], (6, 24), 0, 0.7, (0, 255, 255), 2)
            tiles.append(t)
        for sub in ("training/good", "test/good", "test/defective"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        if tiles:
            cv2.imwrite(str(root / "qa.jpg"), sheet(tiles, cols=10, size=140))
        (root / "skipped.txt").write_text("\n".join(skipped) + "\n")
        print(f"{view:13s} ({camera}, {finder_name}): train {counts['training/good']}, test good "
              f"{counts['test/good']}, test defective {counts['test/defective']}, skipped {len(skipped)}"
              f"  -> {root}/qa.jpg")
    print("\nNo real defective frames yet -> train.py uses synthetic anomalies (indicative only).")


if __name__ == "__main__":
    main()
