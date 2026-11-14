"""LeRobot teleop recordings -> Anomalib training crops (side = main camera, gripper = quick screen).

The VLA demonstrations ("pick up the cube and show the front side to the side camera") already
contain the inspection moment: the arm holds the brick still in front of the side camera. This
script finds that moment from the joint states, decodes those frames and cuts the brick out with
vision/colour_finders.py. The model still trains on photos (crops); video is only the source.

Showing moment = the last still segment (smoothed joint speed < 3, gripper closed < 20, >= 9 frames)
after wrist_flex moved > 30 deg from the start. Frames per episode: --per-episode, spread over it.

Side background: median of frames taken before the showing moment, over ALL episodes (the arm and
brick are elsewhere in each, so the median is the empty scene). Saved as front_background.png.
Gripper: frames recorded while the brick is carried tell what moves (table / hall) behind it.

Faces (dataset/lerobot_faces.csv, episode,face,gripper_face): what the side camera / the gripper camera
sees - top (studs) or bottom (tube); the recordings show one of the two. Empty = unlabelled -> only in
<camera>_all. The file is created with every episode on the first run; fill it from qa.jpg.

Output (gitignored), same layout as build_views.py, kept apart from it (never mixed with booth data):
    <out>/front_top|front_bottom|front_all/{training,test}/good/epNN_fNNNN.png   side camera (= front)
    <out>/gripper_top|gripper_bottom|gripper_all/...                              gripper camera
    <out>/<view>/qa.jpg, skipped.txt, <out>/front_background.png, <out>/episodes.json
Test split: whole episodes held out (--test-frac per brick colour), so near-identical frames of one
episode never sit in both train and test.

Needs pyarrow (LeRobot parquet) and ffmpeg with AV1 (libdav1d):
    uv run --with pyarrow python vision/lerobot_views.py --data ~/Downloads/sorting_robot
    python vision/train_views.py --views dataset/views/lerobot
"""

import argparse
import json
import random
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from colour_finders import gripper_roi_clean, side_colour_bg
from segment import sheet

H, W = 480, 640
FACES = ["top", "bottom", "side"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="LeRobot v3 dataset folder (meta/, data/, videos/)")
    p.add_argument("--out", type=Path, default=Path("dataset/views/lerobot"))
    p.add_argument("--faces", type=Path, default=Path("dataset/lerobot_faces.csv"))
    p.add_argument("--side", default="camera side", help="camera key after observation.images.")
    p.add_argument("--gripper", default="camera gripper")
    p.add_argument("--per-episode", type=int, default=3, help="frames per showing moment")
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-gripper", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------- LeRobot reading
def load(root: Path):
    eps = pd.concat([pd.read_parquet(f) for f in sorted((root / "meta/episodes").rglob("*.parquet"))])
    data = pd.concat([pd.read_parquet(f) for f in sorted((root / "data").rglob("*.parquet"))])
    fps = json.loads((root / "meta/info.json").read_text())["fps"]
    return eps.set_index("episode_index"), data, fps


def showing_segment(state: np.ndarray, fps: int):
    """(a, b) frame range where the brick is held still for the side camera, or None."""
    sp = np.r_[0, np.abs(np.diff(state[:, :5], axis=0)).sum(1)] * fps
    sm = np.convolve(sp, np.ones(9) / 9, "same")
    still = (sm < 3) & (state[:, 5] < 20)
    segs, s = [], None
    for i, v in enumerate(np.r_[still, False]):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if i - s >= 9:
                segs.append((s, i))
            s = None
    cand = [(a, b) for a, b in segs if abs(state[a, 3] - state[0, 3]) > 30]
    return cand[-1] if cand else None


def video(root: Path, row, cam: str):
    k = f"videos/observation.images.{cam}"
    path = root / f"{k}/chunk-{int(row[f'{k}/chunk_index']):03d}/file-{int(row[f'{k}/file_index']):03d}.mp4"
    return path, float(row[f"{k}/from_timestamp"])


def decode(path: Path, t0: float, dur: float, fps: float | None = None) -> np.ndarray:
    """Frames [t0, t0 + dur) as BGR uint8 (N, H, W, 3); fps=None -> native rate."""
    vf = ["-vf", f"fps={fps}"] if fps else []
    raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}",
                          "-i", str(path), *vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def pick(n: int, k: int) -> list[int]:
    """k indexes spread over range(n), away from the ends (the arm settles / starts moving there)."""
    if n <= 0:
        return []
    return sorted({int(round(x)) for x in np.linspace(0.25 * (n - 1), 0.75 * (n - 1), k)})


# ---------------------------------------------------------------- helpers
def colour_of(crop: np.ndarray, mask: np.ndarray) -> str:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[mask > 0]
    hsv = hsv[hsv[:, 1] > 60]
    if len(hsv) == 0:
        return "unknown"
    ang = hsv[:, 0].astype(np.float32) * np.pi / 90
    h = (np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) * 90 / np.pi) % 180
    return "red" if h < 10 or h > 165 else "yellow" if h < 27 else "green" if h < 90 else "other"


def split(colours: dict[int, str], test_frac: float, seed: int) -> dict[int, str]:
    rng = random.Random(seed)
    by = defaultdict(list)
    for e, c in sorted(colours.items()):
        by[c].append(e)
    out = {}
    for c, es in by.items():
        rng.shuffle(es)
        n_test = max(1, round(test_frac * len(es))) if len(es) > 2 else 0
        out |= {e: ("test" if i < n_test else "train") for i, e in enumerate(es)}
    return out


def load_faces(path: Path, episodes) -> dict[int, dict]:
    """episode -> {"face": ..., "gripper_face": ...}; writes a template with every episode if missing."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"episode": sorted(episodes), "face": "", "gripper_face": ""}).to_csv(path, index=False)
        print(f"wrote {path}: fill face / gripper_face (top | bottom) from qa.jpg and re-run")
    df = pd.read_csv(path, dtype=str).fillna("")
    return {int(r.episode): {"face": r.face.strip(), "gripper_face": r.gripper_face.strip()} for r in df.itertuples()}


def main() -> None:
    args = parse_args()
    eps, data, fps = load(args.data)
    side_cam = args.side
    print(f"{len(eps)} episodes, {fps} fps")

    side_frames, grip_out, bg_samples, info = {}, {}, [], {}
    for epi, e in data.groupby("episode_index"):
        epi = int(epi)
        state = np.stack(e["observation.state"].values)
        seg = showing_segment(state, fps)
        if seg is None:
            print(f"ep{epi:02d}: no showing moment, skipped")
            continue
        a, b = seg
        row = eps.loc[epi]
        path, t0 = video(args.data, row, side_cam)
        show = decode(path, t0 + a / fps, (b - a) / fps)
        idx = pick(len(show), args.per_episode)
        side_frames[epi] = [(k, show[k].copy()) for k in idx]
        pre = decode(path, t0, a / fps, fps=max(1.0, 8 * fps / max(a, 1)))  # ~8 frames before the show
        bg_samples += list(pre[:8])
        info[epi] = {"show": [int(a), int(b)], "frames": [int(a + k) for k in idx]}

        if not args.no_gripper:
            opened = np.where(state[:a, 5] > 30)[0]  # carry = closed on the brick after the last open
            c0 = int(opened[-1]) + 10 if len(opened) else 0
            gpath, g0 = video(args.data, row, args.gripper)
            carry = list(decode(gpath, g0 + c0 / fps, max(a - c0, 1) / fps, fps=5)[2:])
            gshow = decode(gpath, g0 + a / fps, (b - a) / fps)
            res = []
            for k in idx:
                k = min(k, len(gshow) - 1)
                if len(carry) < 3:
                    res.append((k, None, None, None, {"ok": False, "reasons": ["too few carry frames"]}))
                    continue
                try:
                    res.append((k, *gripper_roi_clean(gshow[k], carry)))
                except ValueError as ex:
                    res.append((k, None, None, None, {"ok": False, "reasons": [str(ex)]}))
            grip_out[epi] = res
        print(f"ep{epi:02d}: show {a}-{b}, {len(idx)} frames", flush=True)

    background = np.median(np.stack(bg_samples), axis=0).astype(np.uint8)
    args.out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out / "front_background.png"), background)

    # side crops first: they give the brick colour for the per-colour split
    side_res, colours = {}, {}
    for epi, frames in side_frames.items():
        side_res[epi] = []
        for k, img in frames:
            try:
                mask, crop, crop_m, inf = side_colour_bg(img, background)
            except ValueError as ex:
                mask = crop = crop_m = None
                inf = {"ok": False, "reasons": [str(ex)]}
            side_res[epi].append((k, mask, crop, crop_m, inf))
            if crop is not None and inf["ok"] and epi not in colours:
                colours[epi] = colour_of(crop, crop_m)
        colours.setdefault(epi, "unknown")
    sp = split(colours, args.test_frac, args.seed)

    faces = load_faces(args.faces, info)
    summary = []
    for prefix, field, results in (("front", "face", side_res), ("gripper", "gripper_face", grip_out)):
        if not results:
            continue
        views = [f"{prefix}_{f}" for f in FACES] + [f"{prefix}_all"]
        for v in views:
            shutil.rmtree(args.out / v, ignore_errors=True)
            for sub in ("training/good", "test/good"):
                (args.out / v / sub).mkdir(parents=True, exist_ok=True)
        tiles, skipped, n = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(int))
        for epi, res in sorted(results.items()):
            sub = "training/good" if sp[epi] == "train" else "test/good"
            face = faces.get(epi, {}).get(field, "")
            if face in ("none", "occluded"):
                continue
            targets = ([f"{prefix}_{face}"] if face in FACES else []) + [f"{prefix}_all"]
            for k, _, crop, _, inf in res:
                name = f"ep{epi:02d}_f{info[epi]['show'][0] + k:04d}"
                for v in targets:
                    if crop is None or not inf["ok"]:
                        skipped[v].append(f"{name} ({sub}): {', '.join(inf['reasons']) or 'no brick'}")
                        continue
                    cv2.imwrite(str(args.out / v / sub / f"{name}.png"), crop)
                    n[v][sub] += 1
                    t = crop.copy()
                    cv2.putText(t, f"ep{epi:02d} {sub[:5]} {face or '?'}", (6, 24), 0, 0.6, (0, 255, 255), 2)
                    tiles[v].append(t)
        for v in views:
            if not tiles[v]:
                shutil.rmtree(args.out / v, ignore_errors=True)
                continue
            cv2.imwrite(str(args.out / v / "qa.jpg"), sheet(tiles[v], cols=12, size=140))
            (args.out / v / "skipped.txt").write_text("\n".join(skipped[v]) + "\n")
            summary.append(f"{v:15s} train {n[v]['training/good']:3d}  test good {n[v]['test/good']:3d}  "
                           f"skipped {len(skipped[v])}")

    for epi in info:
        info[epi] |= {"colour": colours.get(epi), "split": sp.get(epi), **faces.get(epi, {})}
    (args.out / "episodes.json").write_text(json.dumps(info, indent=1))
    print("\n".join(summary))
    print(f"colours: { {c: list(colours.values()).count(c) for c in set(colours.values())} }")
    print(f"\nLook at {args.out}/<view>/qa.jpg. No defective frames here -> thresholds use synthetic defects. "
          f"Next: python vision/train_views.py --views {args.out}")


if __name__ == "__main__":
    main()
