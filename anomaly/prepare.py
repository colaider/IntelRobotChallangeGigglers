"""Step 1: raw sessions -> cropped Anomalib dataset (+ synthetic defects for testing).

    .venv\\Scripts\\python.exe anomaly/prepare.py            # all sessions in data/

What it does
  * reads the GROUND camera of every session in data/ (colour from the session name)
  * builds the background (median of all frames) and saves it: work/background.png
  * finds + crops the brick in every frame; frames without a brick are listed and skipped
  * per colour, holds out 10 % of the good bricks for testing (seeded, reproducible)
  * draws 6 synthetic marker defects on every held-out brick -> test/defective (+ masks)
  * real defective captures (label=defective), if any, also go to test/defective
  * QA pictures: work/qa_masks.jpg, work/qa_crops.jpg, work/qa_skipped.jpg; work/split.csv
"""

import argparse
import csv
import math
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import synth  # noqa: E402
from brick import BrickFinder, overlay  # noqa: E402


def sheet(images: list[np.ndarray], cols: int = 10, size: int = 140) -> np.ndarray:
    tiles = [cv2.resize(t, (size, size), interpolation=cv2.INTER_AREA) for t in images]
    while len(tiles) % cols:
        tiles.append(np.zeros((size, size, 3), np.uint8))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def collect(data: Path, camera: str) -> list[dict]:
    """Every capture of `camera` in every session: {path, session, id, colour, label, defect}."""
    items = []
    for s in sorted(p for p in data.iterdir() if (p / "manifest.csv").exists()):
        with (s / "manifest.csv").open(newline="") as f:
            for row in csv.DictReader(f):
                path = s / f"{row['id']}_{camera}_color.png"
                if path.exists():
                    items.append({"path": path, "session": s.name, "id": row["id"],
                                  "colour": C.colour_of(s, row.get("color", "")),
                                  "label": row.get("label", "good") or "good", "defect": row.get("defect", "")})
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=C.DATA)
    ap.add_argument("--camera", default=C.CAMERA)
    ap.add_argument("--test-fraction", type=float, default=C.TEST_FRACTION)
    ap.add_argument("--marks", type=int, default=C.MARKS_PER_TEST_IMAGE, help="synthetic defects per test brick")
    args = ap.parse_args()

    items = collect(args.data, args.camera)
    if not items:
        raise SystemExit(f"No '{args.camera}' images in {args.data} (did collection/run.py save that camera?)")
    print(f"{len(items)} '{args.camera}' frames in {len({i['session'] for i in items})} sessions")

    shutil.rmtree(C.DATASET, ignore_errors=True)
    for d in ("training/good", "test/good", "test/defective", "test/masks"):
        (C.DATASET / d).mkdir(parents=True)

    frames = [cv2.imread(str(i["path"])) for i in items]
    finder = BrickFinder.from_frames(frames[:: max(1, len(frames) // 200)])
    finder.save(C.WORK / "background.png")

    # crop every frame
    good_by_colour, qa_mask, qa_crop, qa_skip = defaultdict(list), [], [], []
    rows = []
    for it, img in zip(items, frames):
        m = finder.mask(img)
        name = f"{it['colour']}__{it['session']}_{it['id']}"
        if m is None:
            qa_skip.append(overlay(img, None))
            rows.append({**it, "name": name, "split": "skipped (no brick)"})
            continue
        crop, crop_m = finder.crop(img)
        it.update(name=name, crop=crop, crop_mask=crop_m)
        qa_mask.append(overlay(img, m))
        qa_crop.append(crop)
        if it["label"] == "defective":
            cv2.imwrite(str(C.DATASET / "test/defective" / f"{name}__real_{it['defect'] or 'defect'}.png"), crop)
            rows.append({**it, "split": "test/defective (real)"})
        else:
            good_by_colour[it["colour"]].append(it)

    # split per colour + synthetic marks
    rng = random.Random(C.SEED)
    counts = {}
    for colour, its in sorted(good_by_colour.items()):
        n_test = max(1, math.ceil(args.test_fraction * len(its))) if len(its) > 1 else 0
        test_ids = set(rng.sample(range(len(its)), n_test))
        for k, it in enumerate(its):
            split = "test/good" if k in test_ids else "training/good"
            cv2.imwrite(str(C.DATASET / split / f"{it['name']}.png"), it["crop"])
            rows.append({**it, "split": split})
            if split == "test/good":
                for j in range(args.marks):
                    kind = synth.KINDS[j % len(synth.KINDS)]
                    img_d, m_d = synth.draw(it["crop"], it["crop_mask"], kind, rng)
                    fn = f"{it['name']}__syn{j}_{kind}.png"
                    cv2.imwrite(str(C.DATASET / "test/defective" / fn), img_d)
                    cv2.imwrite(str(C.DATASET / "test/masks" / fn), m_d)
        counts[colour] = (len(its), len(its) - n_test, n_test)

    with (C.WORK / "split.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, ["name", "colour", "session", "id", "label", "split", "path"], extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["colour"], r["session"], r["id"])))
    for fn, imgs in (("qa_masks.jpg", qa_mask), ("qa_crops.jpg", qa_crop), ("qa_skipped.jpg", qa_skip)):
        if imgs:
            cv2.imwrite(str(C.WORK / fn), sheet(imgs))

    print(f"\n{'colour':14s} {'bricks':>6s} {'train':>6s} {'test':>5s}")
    for colour, (n, tr, te) in counts.items():
        print(f"{colour:14s} {n:6d} {tr:6d} {te:5d}")
    n_def = len(list((C.DATASET / 'test/defective').glob('*.png')))
    print(f"no brick (skipped): {len(qa_skip)}   test/defective: {n_def} (synthetic {args.marks} per test brick)")
    print(f"dataset: {C.DATASET}\nQA: {C.WORK / 'qa_crops.jpg'}, {C.WORK / 'qa_skipped.jpg'}")


if __name__ == "__main__":
    main()
