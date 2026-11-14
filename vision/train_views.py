"""Train every model variant from dataset/views/ in one go, export to OpenVINO, calibrate thresholds,
write the model registry and a comparison report.

Per camera (gripper / front, never mixed):
    single    one PatchCore on <camera>_all (every face)
    per_face  one PatchCore per face (<camera>_top / _bottom / _side); at runtime the face is picked by
              model competition (lowest anomaly score = the face the frame belongs to) or by the face
              classifier (train_face_classifier.py, optional)

Output:
    results/models/<view>/...                 checkpoints + OpenVINO IR per view
    results/models/models.json                registry for vision/inspector_bank.py (paths + thresholds)
    results/models/compare.md                 single vs per_face per camera: AUROC, caught@0FA, face accuracy

Thresholds per model: low = highest clean test score + margin; high = halfway to the lowest defective
score (real defective frames if present, else synthetic marker defects drawn on the clean test crops -
indicative only). Re-run after every new dataset.

Example:
    python vision/build_views.py && python vision/train_views.py
    python vision/train_views.py --variants per_face --cameras front --int8
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from benchmark import Scorer
from common import auroc
from export import export
from synth_defects import brick_mask, draw_defect, ink_bgr

FACES = ["top", "bottom", "side"]
CAMERA_INFO = {"gripper": {"camera": "gripper", "finder": "window"}, "front": {"camera": "ground", "finder": "bg"}}
IMG = (".png", ".jpg", ".jpeg")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--views", type=Path, default=Path("dataset/views"))
    p.add_argument("--out", type=Path, default=Path("results/models"))
    p.add_argument("--cameras", nargs="+", choices=list(CAMERA_INFO), default=list(CAMERA_INFO))
    p.add_argument("--variants", nargs="+", choices=["single", "per_face"], default=["single", "per_face"])
    p.add_argument("--min-train", type=int, default=5, help="skip a view with fewer training images")
    p.add_argument("--int8", action="store_true", help="also export INT8 (registry still points to FP16)")
    p.add_argument("--margin", type=float, default=0.03)
    p.add_argument("--device", default="CPU", help="device used for calibration/comparison scoring")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def images(d: Path) -> list[Path]:
    return sorted(p for p in d.glob("*") if p.suffix.lower() in IMG) if d.is_dir() else []


def train_view(view: str, root: Path, out: Path, seed: int) -> Path | None:
    res = out / view
    cmd = [sys.executable, "vision/train.py", "--root", str(root), "--results", str(res),
           "--num-workers", "0", "--seed", str(seed)]
    print(f"  train {view} ...", flush=True)
    log = res / "train.log"
    res.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    ckpts = sorted(res.rglob("model.ckpt"), key=lambda p: p.stat().st_mtime)
    if rc != 0 or not ckpts:
        print(f"  !! training {view} failed, see {log}")
        return None
    return ckpts[-1]


def synthetic_defects(crops: list[Path], per_crop: int, seed: int) -> list[np.ndarray]:
    rng = random.Random(seed)
    out = []
    for p in crops:
        img = cv2.imread(str(p))
        mask = brick_mask(img, 60)
        for k in range(per_crop):
            kind = ["line", "dot", "curve", "tiny", "scribble", "edge"][k % 6]
            ink = ["black", "blue", "red", "tone"][(k + len(out)) % 4]
            out.append(draw_defect(img, mask, kind, ink_bgr(ink, img, mask), rng)[0])
    return out


def score_imgs(sc: Scorer, imgs) -> list[float]:
    vals = []
    for im in imgs:
        bgr = cv2.imread(str(im)) if isinstance(im, Path) else im
        vals.append(float(np.asarray(sc(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).pred_score).ravel()[0]))
    return vals


def thresholds(good: list[float], bad: list[float], margin: float) -> dict:
    low = min(0.95, (max(good) if good else 0.4) + margin)
    if bad and min(bad) > low:
        high, note = low + 0.5 * (min(bad) - low), "separable"
    else:
        high, note = low + 0.15, "not separable" if bad else "no defects: low + 0.15"
    high = min(max(high, low + 0.02), 0.995)  # keep a real uncertain band even on poor data
    return {"low": round(low, 4), "high": round(high, 4), "note": note,
            "good_max": round(max(good), 4) if good else None, "defective_min": round(min(bad), 4) if bad else None}


def caught_at_0fa(good: list[float], bad: list[float]) -> str:
    return f"{sum(b > max(good) for b in bad)}/{len(bad)}" if good and bad else "-"


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reg_path = args.out / "models.json"
    registry = json.loads(reg_path.read_text()) if reg_path.exists() else {"cameras": {}}
    report = ["# Model variants\n", f"Views: `{args.views}`. Defects: real `test/defective` if present, else "
              "synthetic marker defects on the clean test crops (indicative).\n"]

    for cam in args.cameras:
        info = dict(CAMERA_INFO[cam])
        bg = args.views / f"{cam}_background.png"
        if bg.exists():
            dst = args.out / bg.name
            dst.write_bytes(bg.read_bytes())
            info["background"] = str(dst)
        entry = registry["cameras"].get(cam, {}) | info
        views = []
        if "single" in args.variants:
            views.append(("single", None, f"{cam}_all"))
        if "per_face" in args.variants:
            views += [("per_face", f, f"{cam}_{f}") for f in FACES]
        print(f"\n=== {cam}")

        test_good = {f: images(args.views / f"{cam}_{f}" / "test/good") for f in FACES}
        real_bad = {f: images(args.views / f"{cam}_{f}" / "test/defective") for f in FACES}
        synth_bad = {f: (synthetic_defects(test_good[f], 4, args.seed) if not real_bad[f] else []) for f in FACES}
        bad_of = {f: (real_bad[f] or synth_bad[f]) for f in FACES}
        scorers, per_face_entry = {}, {}
        for variant, face, view in views:
            root = args.views / view
            n_train = len(images(root / "training/good"))
            if n_train < args.min_train:
                print(f"  skip {view}: {n_train} training images")
                continue
            ckpt = train_view(view, root, args.out, args.seed)
            if ckpt is None:
                continue
            precs = ["fp16"] + (["int8"] if args.int8 else [])
            xml = export(ckpt, "patchcore", precs, root, args.out / view / "export")["fp16"]
            sc = Scorer(xml, args.device)
            scorers[view] = sc
            faces = [face] if face else FACES
            good = score_imgs(sc, [p for f in faces for p in test_good[f]])
            bad = score_imgs(sc, [b for f in faces for b in bad_of[f]])
            th = thresholds(good, bad, args.margin)
            e = {"view": view, "xml": str(xml), "n_train": n_train, **th}
            print(f"  {view}: low {th['low']} high {th['high']} ({th['note']})")
            if variant == "single":
                entry["single"] = e
            else:
                per_face_entry[face] = e
        if per_face_entry:
            entry["per_face"] = per_face_entry
        entry.setdefault("classifier", None)
        registry["cameras"][cam] = entry

        # ---- comparison on the same test frames: good of every face vs defects of every face
        rows = []
        lab_good = [(f, p) for f in FACES for p in test_good[f]]
        lab_bad = [(f, b) for f in FACES for b in bad_of[f]]
        if not lab_good or not lab_bad:
            report.append(f"## {cam}\n\nno test frames, comparison skipped\n")
            continue
        if f"{cam}_all" in scorers:
            g = score_imgs(scorers[f"{cam}_all"], [p for _, p in lab_good])
            b = score_imgs(scorers[f"{cam}_all"], [x for _, x in lab_bad])
            rows.append(("single", auroc([0] * len(g) + [1] * len(b), g + b), caught_at_0fa(g, b), "-"))
        face_models = {f: scorers[f"{cam}_{f}"] for f in FACES if f"{cam}_{f}" in scorers}
        if face_models:
            def compete(items):
                scores, right = [], 0
                for f, im in items:
                    per = {ff: score_imgs(s, [im])[0] for ff, s in face_models.items()}
                    win = min(per, key=per.get)
                    right += win == f
                    scores.append(per[win])
                return scores, right
            g, rg = compete(lab_good)
            b, rb = compete(lab_bad)
            acc = f"{rg + rb}/{len(lab_good) + len(lab_bad)}"
            rows.append((f"per_face ({'+'.join(face_models)})", auroc([0] * len(g) + [1] * len(b), g + b),
                         caught_at_0fa(g, b), acc))
        report.append(f"## {cam}\n\ntest: {len(lab_good)} good, {len(lab_bad)} defective "
                      f"({'real' if any(real_bad.values()) else 'synthetic'})\n")
        report.append("| variant | AUROC | caught@0FA | face picked right |\n|---|---|---|---|")
        report += [f"| {v} | {a:.3f} | {c} | {acc} |" for v, a, c, acc in rows]
        report.append("")

    reg_path.write_text(json.dumps(registry, indent=2))
    (args.out / "compare.md").write_text("\n".join(report) + "\n")
    print("\n" + "\n".join(report))
    print(f"\nRegistry: {reg_path}\nReport:   {args.out / 'compare.md'}")


if __name__ == "__main__":
    main()
