"""Compare inspection variants on the raw collection sessions in data/ (one command, one table).

A variant = camera x finder (how the brick is cropped) x grouping (which frames share a model)
x model x CLAHE. Every variant sees the same captures, the same train/test split and the same
synthetic defects, so the numbers are comparable.

    python vision/experiment.py --qa                       # crop QA sheets only (no training)
    python vision/experiment.py                            # default grid (ground + gripper)
    python vision/experiment.py --cameras ground --finders bg scene bg_scene --groupings all per_face

Steps
  1. captures from data/ (vision/sessions.py) + face labels (dataset/faces.csv). ground: only
     frames labelled top/bottom/side; gripper: every frame (the brick is always between the jaws)
  2. split per (colour, face) by arm pose group (near-duplicate frames never straddle the split),
     ~25 % of the pose groups -> test
  3. synthetic marker defects are drawn on the RAW test frames, inside the brick (reference mask
     = the scene finder's mask), so each finder must also find and crop the defective brick
  4. crops are cached per camera x finder (results/experiments/cache/), QA sheet per finder
  5. per variant and group: train on good train crops, score test good + defective crops
  6. results/experiments/<run>/summary.csv + report.md + heatmap galleries

Metrics (per group, then pooled over groups - raw scores of different models are not comparable):
  AUROC        threshold-free separability of good vs defective test crops
  caught@0FA   defective crops scoring above EVERY good test crop of the same group
  found        frames where the finder returned a brick; ok = mask passed the quality check
  crop_ms      median time to find + crop the brick; model_ms = torch CPU per image (OpenVINO
               numbers: vision/benchmark.py on the exported winner)
"""

import argparse
import csv
import json
import random
import time
import zlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from common import auroc
from finders import FINDERS, NEEDS_BACKGROUND, NEEDS_WINDOW, WINDOWS, FindResult, make_finder, median_background
from segment import FILL, sheet
from sessions import Capture, load_captures
from synth_defects import draw_defect, ink_bgr

ROOT = Path("results/experiments")
KINDS = ["line", "curve", "dot", "tiny", "scribble", "edge"]
DEFECT_INKS = ["black", "blue", "red", "tone"]
INSPECT_FACES = {"top", "bottom", "side"}
SIZE = 256
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


# ---------------------------------------------------------------- data + split
def usable(caps: list[Capture], camera: str) -> list[Capture]:
    if camera == "ground":
        return [c for c in caps if c.face in INSPECT_FACES]
    return [c for c in caps if c.face != "occluded"]


def group_of(c: Capture, grouping: str, camera: str) -> str:
    face = c.face if camera == "ground" else "any"  # ground labels say nothing about the gripper view
    return {"all": "all", "per_face": face, "per_colour": c.colour,
            "per_face_colour": f"{face}/{c.colour}"}[grouping]


def split(caps: list[Capture], camera: str, test_frac: float, seed: int) -> dict[str, str]:
    """capture key -> 'train' | 'test'. Stratified per (colour, face), whole pose groups move."""
    rng = random.Random(seed)
    strata = defaultdict(lambda: defaultdict(list))
    for c in caps:
        strata[(c.colour, c.face if camera == "ground" else "any")][c.pose_group].append(c)
    out = {}
    for key in sorted(strata):
        groups = sorted(strata[key])
        rng.shuffle(groups)
        n_test = max(1, round(test_frac * len(groups))) if len(groups) > 2 else 0
        for i, g in enumerate(groups):
            for c in strata[key][g]:
                out[c.key] = "test" if i < n_test else "train"
    return out


# ---------------------------------------------------------------- synthetic defects on raw frames
def defect_frame(bgr: np.ndarray, mask: np.ndarray, kind: str, ink: str, rng: random.Random):
    """Draw one marker defect inside the brick mask, sized relative to the brick (its bbox)."""
    x, y, w, h = cv2.boundingRect(cv2.findNonZero(mask))
    sub, sub_m = bgr[y:y + h, x:x + w].copy(), mask[y:y + h, x:x + w]
    sub_d, sub_dm = draw_defect(sub, sub_m, kind, ink_bgr(ink, sub, sub_m), rng)
    out, dm = bgr.copy(), np.zeros(mask.shape, np.uint8)
    out[y:y + h, x:x + w], dm[y:y + h, x:x + w] = sub_d, sub_dm
    return out, dm


# ---------------------------------------------------------------- crop cache
class CropCache:
    """Crops per camera x finder on disk; frames are given as (name, loader) so nothing is re-read."""

    def __init__(self, camera: str, finder: str, background: np.ndarray | None):
        self.dir = ROOT / "cache" / f"{camera}_{finder}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.find = make_finder(finder, camera, background)

    def get(self, name: str, load) -> tuple[np.ndarray | None, dict]:
        meta_p, img_p = self.dir / f"{name}.json", self.dir / f"{name}.png"
        if meta_p.exists():
            meta = json.loads(meta_p.read_text())
            return (cv2.imread(str(img_p)) if meta["found"] else None), meta
        r: FindResult = self.find(load())
        meta = {"found": r.crop is not None, "ok": r.ok, "reasons": r.reasons, "ms": round(r.ms, 1)}
        if r.crop is not None:
            cv2.imwrite(str(img_p), r.crop)
            cv2.imwrite(str(self.dir / f"{name}_mask.png"), r.crop_mask * 255)
        meta_p.write_text(json.dumps(meta))
        return (cv2.imread(str(img_p)) if meta["found"] else None), meta

    def mask(self, name: str) -> np.ndarray | None:
        m = cv2.imread(str(self.dir / f"{name}_mask.png"), cv2.IMREAD_GRAYSCALE)
        return None if m is None else (m > 127).astype(np.uint8)


def clahe(bgr: np.ndarray) -> np.ndarray:
    """CLAHE on lightness inside the brick; the flat grey background stays flat."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    out[(bgr == FILL).all(axis=2)] = FILL
    return out


# ---------------------------------------------------------------- models (torch, CPU, no Lightning)
def to_tensor(crops: list[np.ndarray]) -> torch.Tensor:
    x = np.stack([cv2.cvtColor(cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                  for c in crops]).astype(np.float32) / 255.0
    return torch.from_numpy(((x - MEAN) / STD).transpose(0, 3, 1, 2).copy())


def build_model(name: str, train_crops: list[np.ndarray], seed: int):
    from anomalib.models import Padim, Patchcore

    torch.manual_seed(seed)
    model = Patchcore(coreset_sampling_ratio=0.1) if name == "patchcore" else Padim()
    model.model.train()
    with torch.no_grad():
        for i in range(0, len(train_crops), 8):
            model.model(to_tensor(train_crops[i:i + 8]))
    model.fit()
    model.model.eval()
    return model


def score(model, crops: list[np.ndarray]) -> tuple[list[float], list[np.ndarray], float]:
    scores, maps, t = [], [], time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(crops), 8):
            out = model.model(to_tensor(crops[i:i + 8]))
            scores += [float(s) for s in out.pred_score.flatten()]
            maps += [m.squeeze().numpy() for m in out.anomaly_map]
    return scores, maps, (time.perf_counter() - t) * 1000 / max(len(crops), 1)


def heat(crop: np.ndarray, amap: np.ndarray, vmax: float) -> np.ndarray:
    c = cv2.resize(crop, (SIZE, SIZE))
    a = np.clip(cv2.resize(amap, (SIZE, SIZE)) / max(vmax, 1e-6), 0, 1)
    return cv2.addWeighted(c, 0.55, cv2.applyColorMap((a * 255).astype(np.uint8), cv2.COLORMAP_JET), 0.45, 0)


def gallery(items: list[tuple[np.ndarray, str]], cols: int = 8, size: int = 180) -> np.ndarray:
    tiles = []
    for img, text in items:
        t = cv2.resize(img, (size, size))
        cv2.rectangle(t, (0, 0), (size, 18), (0, 0, 0), -1)
        cv2.putText(t, text, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.zeros((size, size, 3), np.uint8))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


# ---------------------------------------------------------------- main
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--cameras", nargs="+", default=["ground", "gripper"])
    p.add_argument("--finders", nargs="+", choices=FINDERS, default=FINDERS)
    p.add_argument("--groupings", nargs="+", default=["all", "per_face"],
                   choices=["all", "per_face", "per_colour", "per_face_colour"])
    p.add_argument("--models", nargs="+", default=["patchcore"], choices=["patchcore", "padim"])
    p.add_argument("--clahe", nargs="+", default=["off"], choices=["off", "on"])
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--defects-per-frame", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--qa", action="store_true", help="only crop + QA sheets, no training")
    p.add_argument("--name", default=None, help="run folder name (default: date_time)")
    return p.parse_args()


def compatible(camera: str, finder: str) -> bool:
    if finder in NEEDS_WINDOW and camera not in WINDOWS:
        return False
    if finder in NEEDS_BACKGROUND and camera != "ground":  # only the ground camera stands still
        return False
    return True


def main() -> None:
    args = parse_args()
    run = ROOT / (args.name or time.strftime("%Y%m%d_%H%M%S"))
    run.mkdir(parents=True, exist_ok=True)
    all_caps = load_captures(args.data)
    if not any(c.face for c in all_caps):
        raise SystemExit("No face labels: fill dataset/faces.csv first (python vision/sessions.py --sheet ground)")
    rows, qa_lines = [], []

    for camera in args.cameras:
        caps = usable(all_caps, camera)
        sp = split(caps, camera, args.test_frac, args.seed)
        test_caps = [c for c in caps if sp[c.key] == "test"]
        print(f"\n=== {camera}: {len(caps)} frames, test {len(test_caps)}")

        background = None
        if camera == "ground":  # fixed camera: empty scene = median of every frame, brick moving
            background = median_background([c.image(camera) for c in all_caps])
            cv2.imwrite(str(run / f"{camera}_background.png"), background)

        # reference brick masks for drawing defects: the cleanest finder per camera (crop QA)
        ref = make_finder("bg", camera, background) if camera == "ground" else make_finder("window", camera)
        defects = []  # (name, capture, kind, ink, defective frame path)
        for c in test_caps:
            bgr = c.image(camera)
            r = ref(bgr)
            if r.frame_mask is None or not r.ok:
                continue
            for j in range(args.defects_per_frame):
                kind, ink = KINDS[j % len(KINDS)], DEFECT_INKS[(j + len(defects)) % len(DEFECT_INKS)]
                name = f"{c.key}__syn{j}_{kind}_{ink}"
                d_dir = ROOT / "cache" / f"{camera}_defective_frames"
                d_dir.mkdir(parents=True, exist_ok=True)
                p = d_dir / f"{name}.png"
                if not p.exists():
                    img_d, _ = defect_frame(bgr, r.frame_mask, kind, ink, random.Random(zlib.crc32(name.encode())))
                    cv2.imwrite(str(p), img_d)
                defects.append((name, c, kind, ink, p))
        print(f"{len(defects)} synthetic defective frames from {len({d[1].key for d in defects})} test frames")

        for finder in args.finders:
            if not compatible(camera, finder):
                continue
            cache = CropCache(camera, finder, background)
            crops, metas = {}, {}
            for c in caps:
                crops[c.key], metas[c.key] = cache.get(c.key, lambda c=c: c.image(camera))
            dcrops = {}
            for name, c, kind, ink, p in defects:
                dcrops[name], metas[name] = cache.get(name, lambda p=p: cv2.imread(str(p)))
            masks = {k: cache.mask(k) for k in [*crops, *dcrops]}
            found = [k for k in crops if crops[k] is not None]
            ok = [k for k in found if metas[k]["ok"]]
            crop_ms = float(np.median([metas[c.key]["ms"] for c in caps if metas[c.key]["ms"] > 0] or [0]))
            qa = [crops[k] for k in sorted(found)]
            if qa:
                cv2.imwrite(str(run / f"qa_{camera}_{finder}.jpg"), sheet(qa, cols=10, size=128))
            line = (f"{camera:8s} {finder:10s} found {len(found)}/{len(caps)}  ok {len(ok)}/{len(caps)}  "
                    f"defective found {sum(v is not None for v in dcrops.values())}/{len(defects)}  "
                    f"crop {crop_ms:.0f} ms")
            print(line)
            qa_lines.append(line)
            if args.qa:
                continue

            for grouping in args.groupings:
                if grouping in ("per_face", "per_face_colour") and camera != "ground":
                    continue
                for model_name in args.models:
                    for cl in args.clahe:
                        prep = clahe if cl == "on" else (lambda x: x)
                        res = evaluate(caps, sp, crops, dcrops, masks, defects, camera, grouping, model_name, prep,
                                       args.seed, run / f"{camera}_{finder}_{grouping}_{model_name}_clahe-{cl}")
                        res.update(camera=camera, finder=finder, grouping=grouping, model=model_name, clahe=cl,
                                   frames=len(caps), found=len(found), ok=len(ok), crop_ms=round(crop_ms))
                        rows.append(res)
                        print(f"   {grouping:15s} {model_name:9s} clahe {cl:3s}  AUROC {res['auroc']:.3f} "
                              f"(brick {res['b_auroc']:.3f})  caught@0FA {res['caught']} (brick {res['b_caught']})  "
                              f"paired {res['paired']:.2f}  model {res['model_ms']:.0f} ms")

    (run / "crop_qa.txt").write_text("\n".join(qa_lines) + "\n")
    if rows:
        write_report(run, rows, args)
    print(f"\n-> {run}")


def brick_score(amap: np.ndarray, mask: np.ndarray | None, erode: float = 0.04) -> float:
    """Max of the anomaly map ON the brick (eroded crop mask): leftovers of gripper / background
    at the crop border cannot raise the score, a marker on the brick still does."""
    if mask is None:
        return float(amap.max())
    m = cv2.resize(mask, amap.shape[::-1], interpolation=cv2.INTER_NEAREST)
    k = max(3, int(erode * amap.shape[0])) | 1
    m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return float(amap[m > 0].max()) if m.any() else float(amap.max())


class Tally:
    """Metrics of one scoring rule, pooled over the groups of a variant."""

    def __init__(self):
        self.auc, self.caught, self.n_def, self.paired, self.kinds = [], 0, 0, 0, defaultdict(lambda: [0, 0])

    def add(self, sg: dict[str, float], sb: list[tuple[str, str, str, float]]) -> tuple[float, int, float]:
        """sg: good key -> score; sb: (defect name, kind, source key, score)."""
        good, bad = list(sg.values()), [s for *_, s in sb]
        a = auroc([0] * len(good) + [1] * len(bad), good + bad)
        top = max(good)
        caught = sum(s > top for s in bad)
        self.auc.append((a, len(good) + len(bad)))
        self.caught, self.n_def = self.caught + caught, self.n_def + len(bad)
        self.paired += sum(s > sg[src] for _, _, src, s in sb if src in sg)
        for _, kind, _, s in sb:
            self.kinds[kind][0] += s > top
            self.kinds[kind][1] += 1
        return a, caught, top

    def result(self, prefix: str = "") -> dict:
        w = sum(n for _, n in self.auc)
        out = {"auroc": round(sum(a * n for a, n in self.auc) / w, 3) if w else float("nan"),
               "caught": f"{self.caught}/{self.n_def}",
               "caught_frac": round(self.caught / self.n_def, 3) if self.n_def else 0.0,
               "paired": round(self.paired / self.n_def, 3) if self.n_def else 0.0}
        if not prefix:
            out.update({f"k_{k}": f"{v[0]}/{v[1]}" for k, v in sorted(self.kinds.items())})
        return {prefix + k: v for k, v in out.items()}


def evaluate(caps, sp, crops, dcrops, masks, defects, camera, grouping, model_name, prep, seed, out: Path) -> dict:
    groups = sorted({group_of(c, grouping, camera) for c in caps})
    t_model, t_brick, n_good = Tally(), Tally(), 0
    per_group, model_ms, good_tiles, bad_tiles = {}, [], [], []
    for g in groups:
        gc = [c for c in caps if group_of(c, grouping, camera) == g]
        train = [prep(crops[c.key]) for c in gc if sp[c.key] == "train" and crops[c.key] is not None]
        tgood = [(c.key, prep(crops[c.key])) for c in gc if sp[c.key] == "test" and crops[c.key] is not None]
        gkeys = {c.key for c in gc}
        tbad = [(n, kind, c.key, prep(dcrops[n])) for n, c, kind, ink, _ in defects
                if c.key in gkeys and dcrops[n] is not None]
        if len(train) < 3 or not tgood or not tbad:
            per_group[g] = "skipped (too few frames)"
            continue
        model = build_model(model_name, train, seed)
        sg, mg, ms = score(model, [x for _, x in tgood])
        sb, mb, _ = score(model, [x for *_, x in tbad])
        model_ms.append(ms)
        bg_ = [brick_score(m, masks.get(k)) for (k, _), m in zip(tgood, mg)]
        bb_ = [brick_score(m, masks.get(n)) for (n, *_), m in zip(tbad, mb)]
        a, caught, top = t_model.add({k: s for (k, _), s in zip(tgood, sg)},
                                     [(n, kind, src, s) for (n, kind, src, _), s in zip(tbad, sb)])
        ab, caught_b, _ = t_brick.add({k: s for (k, _), s in zip(tgood, bg_)},
                                      [(n, kind, src, s) for (n, kind, src, _), s in zip(tbad, bb_)])
        n_good += len(sg)
        per_group[g] = (f"AUROC {a:.3f} (on brick {ab:.3f}), caught {caught}/{len(sb)} (on brick {caught_b}), "
                        f"train {len(train)}, test good {len(sg)}")
        vmax = max(sb + sg)
        good_tiles += [(heat(x, m, vmax), f"{g} {s:.2f}") for (_, x), m, s in zip(tgood, mg, sg)]
        bad_tiles += [(heat(x, m, vmax), f"{kind} {s:.2f}{'' if s > top else ' MISS'}")
                      for (_, kind, _, x), m, s in zip(tbad, mb, sb)]
    out.mkdir(parents=True, exist_ok=True)
    if good_tiles:
        cv2.imwrite(str(out / "test_good.jpg"), gallery(good_tiles))
        cv2.imwrite(str(out / "test_defective.jpg"), gallery(bad_tiles[:64]))
    (out / "groups.json").write_text(json.dumps(per_group, indent=1))
    return {**t_model.result(), **t_brick.result("b_"), "test_good": n_good,
            "model_ms": round(float(np.mean(model_ms)) if model_ms else 0.0, 1), "groups": json.dumps(per_group)}


def write_report(run: Path, rows: list[dict], args) -> None:
    keys = ["camera", "finder", "grouping", "model", "clahe", "auroc", "caught", "caught_frac", "paired",
            "b_auroc", "b_caught", "b_caught_frac", "b_paired", "frames", "found", "ok", "test_good", "crop_ms",
            "model_ms"] + [f"k_{k}" for k in KINDS] + ["groups"]
    with (run / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    rows = sorted(rows, key=lambda r: (-max(r["caught_frac"], r["b_caught_frac"]), -r["auroc"]))
    lines = [f"# Experiment {run.name}", "", f"Args: `{' '.join(f'--{k} {v}' for k, v in vars(args).items())}`", "",
             "- caught@0FA: defective crops scoring above EVERY good test crop of the same model",
             "- paired: defective crop scores above the same frame without the defect (detection signal "
             "without pose-to-pose variation)",
             "- model = PatchCore image score; brick = max of the anomaly map on the eroded brick mask",
             "- per-kind columns: caught@0FA with the model score", "",
             "| camera | finder | grouping | model | clahe | AUROC | caught@0FA | paired | AUROC brick | caught brick "
             "| paired brick | found | ok | crop ms | model ms | " + " | ".join(KINDS) + " |",
             "|" + "---|" * (15 + len(KINDS))]
    for r in rows:
        lines.append(f"| {r['camera']} | {r['finder']} | {r['grouping']} | {r['model']} | {r['clahe']} | {r['auroc']} "
                     f"| {r['caught']} | {r['paired']} | {r['b_auroc']} | {r['b_caught']} | {r['b_paired']} "
                     f"| {r['found']}/{r['frames']} | {r['ok']}/{r['frames']} | {r['crop_ms']} | "
                     f"{r['model_ms']} | " + " | ".join(r.get(f"k_{k}", "") for k in KINDS) + " |")
    lines += ["", "Per variant: `<camera>_<finder>_<grouping>_<model>_clahe-<x>/` has test_good.jpg, "
              "test_defective.jpg (heatmaps, MISS = below the best-scoring good crop) and groups.json.",
              "Crop QA: `qa_<camera>_<finder>.jpg`, counts in `crop_qa.txt`."]
    (run / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
