"""Pick the two decision thresholds (low / high) for inspect_brick.py from real frames of THIS setup.

    score <  low   -> good
    score >= high  -> defective
    in between     -> uncertain (re-shoot the face)

low  = highest score of the clean frames + a margin, so clean bricks pass.
high = with defective frames: halfway between low and the lowest defective score (all of them
       still count as defective); without: low + band.
A false "defective" ends a brick's inspection early (judges' rule), so `high` is deliberately
above the clean scores with an uncertain band in between.

Use frames that were NOT used for training (e.g. dataset/raw/test). Re-run after any change of
camera, light, background or model.

Example:
    python vision/calibrate.py --xml results/export/patchcore/fp16/weights/openvino/model.xml \
        --good dataset/raw/test/good --defective dataset/raw/test/defective --out results/calibration.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from inspect_brick import Inspector
from segment import IMAGE_EXT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xml", type=Path, required=True)
    p.add_argument("--good", type=Path, required=True, help="clean frames (not used in training)")
    p.add_argument("--defective", type=Path, default=None, help="defective frames (optional)")
    p.add_argument("--seg", choices=["scene", "paper", "none"], default="scene")
    p.add_argument("--device", default="CPU")
    p.add_argument("--margin", type=float, default=0.03, help="added to the highest clean score")
    p.add_argument("--band", type=float, default=0.15, help="uncertain band when no defective frames")
    p.add_argument("--out", type=Path, default=Path("results/calibration.json"))
    return p.parse_args()


def scores(insp: Inspector, folder: Path) -> tuple[list[float], list[str]]:
    vals, skipped = [], []
    for p in sorted(q for q in folder.rglob("*") if q.suffix.lower() in IMAGE_EXT):
        r = insp.inspect_face(cv2.imread(str(p)), p.stem)
        if r["score"] is None:
            skipped.append(f"{p.name}: {r['reason']}")
        else:
            vals.append(r["score"])
    return vals, skipped


def main() -> None:
    args = parse_args()
    insp = Inspector(args.xml, args.device, None if args.seg == "none" else args.seg, low=0.0, high=1.01)
    good, skip_g = scores(insp, args.good)
    if not good:
        raise SystemExit("no clean frame could be scored")
    low = min(0.95, max(good) + args.margin)
    result = {"low": round(low, 4), "n_good": len(good), "good_max": round(max(good), 4),
              "good_p50": round(float(np.median(good)), 4)}
    if args.defective:
        bad, skip_d = scores(insp, args.defective)
        result.update(n_defective=len(bad))
        if bad:
            result.update(defective_min=round(min(bad), 4), defective_p50=round(float(np.median(bad)), 4))
            if min(bad) > low:
                high = low + 0.5 * (min(bad) - low)
                result["note"] = "separable: every defective frame >= high"
            else:
                high = low + args.band
                missed = sum(b < low for b in bad)
                result["note"] = (f"NOT separable: {missed}/{len(bad)} defective frames score below low "
                                  f"(would pass as good) - more data / better light needed")
        else:
            high = low + args.band
        skip_g += skip_d
    else:
        high = low + args.band
        result["note"] = "no defective frames: high = low + band (indicative)"
    result["high"] = round(min(high, 0.99), 4)
    result["skipped_uncertain_segmentation"] = skip_g
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
