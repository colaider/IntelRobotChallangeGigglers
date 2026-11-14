"""Step 3: evaluate the saved model and write a report folder.

    .venv\\Scripts\\python.exe anomaly/evaluate.py                 # CPU
    .venv\\Scripts\\python.exe anomaly/evaluate.py --devices CPU GPU NPU   # on the Intel laptop

reports/<date_time>/
    report.md            summary: data, metrics, per defect kind, per colour, speed, pictures
    scores.csv           every test image: colour, kind, true label, score, verdict
    frames.csv           every raw frame through the FULL pipeline (brick finding + model)
    histogram.png        score distribution good vs defective, threshold line
    gallery_good.jpg     all held-out good bricks with heatmaps
    gallery_defective.jpg  synthetic defects with heatmaps (worst-scored first)
    heatmaps/            one picture per test image
    qa_*.jpg             brick-finding pictures from prepare.py

Uses the same Inspector as the robot will, so the numbers are what the robot gets.
"""

import argparse
import csv
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from inspector import Inspector, heat_overlay  # noqa: E402


def auroc(labels, scores) -> float:
    labels, scores = np.asarray(labels), np.asarray(scores, float)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def best_f1_threshold(labels, scores) -> tuple[float, float]:
    best = (0.5, 0.0)
    for t in sorted(set(scores)):
        pred = np.asarray(scores) >= t
        lab = np.asarray(labels).astype(bool)
        tp, fp, fn = (pred & lab).sum(), (pred & ~lab).sum(), (~pred & lab).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best[1]:
            best = (float(t), float(f1))
    return best


def gallery(items: list[tuple[np.ndarray, str]], cols: int = 6, size: int = 200) -> np.ndarray:
    tiles = []
    for img, text in items:
        t = cv2.resize(img, (size, size))
        cv2.rectangle(t, (0, 0), (size, 20), (0, 0, 0), -1)
        cv2.putText(t, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.zeros((size, size, 3), np.uint8))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def kind_of(name: str) -> str:
    if "__syn" in name:
        return name.split("__syn")[1].split("_", 1)[1].rsplit(".", 1)[0]
    if "__real_" in name:
        return "real_" + name.split("__real_")[1].rsplit(".", 1)[0]
    return "good"


def histogram(path: Path, good: list[float], bad: list[float], thr: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=120)
    bins = np.linspace(0, 1, 26)
    ax.hist(np.clip(good, 0, 1), bins, alpha=0.7, label=f"good ({len(good)})", color="#2a9d8f")
    ax.hist(np.clip(bad, 0, 1), bins, alpha=0.7, label=f"defective ({len(bad)})", color="#e76f51")
    ax.axvline(thr, color="k", ls="--", lw=1, label=f"threshold {thr:.2f}")
    ax.set_xlabel("anomaly score")
    ax.set_ylabel("images")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", nargs="+", default=["CPU"])
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    out = C.REPORTS / time.strftime("%Y%m%d_%H%M%S")
    (out / "heatmaps").mkdir(parents=True)
    insp = Inspector(device=args.devices[0], threshold=args.threshold)
    thr = insp.threshold

    # ---- 1. test crops ---------------------------------------------------------
    rows = []
    for split, label in (("test/good", 0), ("test/defective", 1)):
        for p in sorted((C.DATASET / split).glob("*.png")):
            crop = cv2.imread(str(p))
            score, amap = insp.score_crop(crop)
            heat = heat_overlay(cv2.resize(crop, (insp.size, insp.size)), amap)
            cv2.imwrite(str(out / "heatmaps" / p.name.replace(".png", ".jpg")), np.hstack(
                [cv2.resize(crop, (insp.size, insp.size)), heat]))
            rows.append({"image": p.name, "colour": p.name.split("__")[0], "kind": kind_of(p.name), "label": label,
                         "score": round(score, 4), "verdict": "defective" if score >= thr else "good",
                         "heat": heat})
    labels = [r["label"] for r in rows]
    scores = [r["score"] for r in rows]
    good = [r for r in rows if r["label"] == 0]
    bad = [r for r in rows if r["label"] == 1]
    tp = sum(r["verdict"] == "defective" for r in bad)
    fp = sum(r["verdict"] == "defective" for r in good)
    best_t, best_f1 = best_f1_threshold(labels, scores)
    summary = {
        "auroc": round(auroc(labels, scores), 3),
        "threshold": thr,
        "defects_caught": f"{tp}/{len(bad)}",
        "good_flagged": f"{fp}/{len(good)}",
        "f1_at_threshold": round(2 * tp / max(2 * tp + fp + (len(bad) - tp), 1), 3),
        "best_f1_threshold": round(best_t, 3), "best_f1": round(best_f1, 3),
    }

    by_kind, by_colour = defaultdict(list), defaultdict(list)
    for r in bad:
        by_kind[r["kind"]].append(r)
    for r in rows:
        by_colour[r["colour"]].append(r)
    gscores = [r["score"] for r in good]
    kind_rows = [(k, f"{sum(r['verdict'] == 'defective' for r in v)}/{len(v)}",
                  round(auroc([0] * len(gscores) + [1] * len(v), gscores + [r["score"] for r in v]), 3),
                  round(float(np.mean([r["score"] for r in v])), 3)) for k, v in sorted(by_kind.items())]
    colour_rows = []
    for c, v in sorted(by_colour.items()):
        g, d = [r for r in v if r["label"] == 0], [r for r in v if r["label"] == 1]
        colour_rows.append((c, len(g), len(d), round(auroc([r["label"] for r in v], [r["score"] for r in v]), 3),
                            f"{sum(r['verdict'] == 'good' for r in g)}/{len(g)}",
                            f"{sum(r['verdict'] == 'defective' for r in d)}/{len(d)}"))

    with (out / "scores.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, ["image", "colour", "kind", "label", "score", "verdict"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    histogram(out / "histogram.png", gscores, [r["score"] for r in bad], thr)
    cv2.imwrite(str(out / "gallery_good.jpg"),
                gallery([(r["heat"], f"{r['colour']} {r['score']:.2f}") for r in good]))
    worst = sorted(bad, key=lambda r: r["score"])
    cv2.imwrite(str(out / "gallery_defective.jpg"),
                gallery([(r["heat"], f"{r['kind']} {r['score']:.2f}") for r in worst[:36]]))

    # ---- 2. full pipeline on every raw frame (brick finding + model, timing) ------
    frames = []
    with (C.WORK / "split.csv").open(newline="") as f:
        split_rows = list(csv.DictReader(f))
    for s in split_rows:
        img = cv2.imread(s["path"])
        r = insp.inspect(img, make_view=False)
        frames.append({"frame": s["path"], "colour": s["colour"], "split": s["split"], "result": r.label,
                       "score": "" if r.score is None else round(r.score, 4), "ms": round(r.ms, 1)})
    with (out / "frames.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, list(frames[0]))
        w.writeheader()
        w.writerows(frames)
    ms = [fr["ms"] for fr in frames if fr["result"] != "no_brick"]
    found = sum(fr["result"] != "no_brick" for fr in frames)

    # ---- 3. speed per device -------------------------------------------------------
    speed = []
    sample = cv2.imread(split_rows[0]["path"])
    for dev in args.devices:
        try:
            d_insp = insp if dev == args.devices[0] else Inspector(device=dev, threshold=thr)
        except RuntimeError as e:
            speed.append((dev, str(e), "", ""))
            continue
        crop = cv2.imread(str(next((C.DATASET / "test/good").glob("*.png"))))
        t_model = []
        for _ in range(30):
            t = time.perf_counter()
            d_insp.score_crop(crop)
            t_model.append((time.perf_counter() - t) * 1000)
        t_full = []
        for _ in range(10):
            t_full.append(d_insp.inspect(sample, make_view=False).ms)
        speed.append((dev, "ok", f"{np.median(t_model):.0f}", f"{np.median(t_full):.0f}"))

    for q in ("qa_crops.jpg", "qa_masks.jpg", "qa_skipped.jpg"):
        if (C.WORK / q).exists():
            shutil.copy2(C.WORK / q, out / q)

    # ---- 4. report -------------------------------------------------------------------
    info = insp.info
    lines = [
        f"# Anomaly report - {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"Model: {info['model']} ({info['backbone']}), {info['precision']}, input {info['image_size']} px, "
        f"trained {info['created']} on {info['counts']['training/good']} good bricks "
        f"({', '.join(info['colours'])}), camera `{info['camera']}`.",
        "",
        "Test set = 10 % of the good bricks of every colour (never seen in training) + synthetic marker "
        "defects drawn on exactly those bricks.",
        "",
        "## Result",
        "",
        "| metric | value |", "|---|---|",
        f"| AUROC (good vs defective, threshold-free) | **{summary['auroc']}** |",
        f"| defects caught at threshold {thr:.2f} | {summary['defects_caught']} |",
        f"| good bricks wrongly flagged | {summary['good_flagged']} |",
        f"| F1 at threshold | {summary['f1_at_threshold']} |",
        f"| best-F1 threshold on this test set | {summary['best_f1_threshold']} (F1 {summary['best_f1']}) |",
        "",
        "![histogram](histogram.png)",
        "",
        "## Per defect kind (AUROC vs all held-out good bricks)",
        "",
        "| kind | caught | AUROC | mean score |", "|---|---|---|---|",
        *[f"| {k} | {c} | {a} | {m} |" for k, c, a, m in kind_rows],
        "",
        "## Per colour",
        "",
        "| colour | good | defective | AUROC | good passed | defects caught |", "|---|---|---|---|---|---|",
        *[f"| {c} | {g} | {d} | {a} | {gp} | {dc} |" for c, g, d, a, gp, dc in colour_rows],
        "",
        "## Full pipeline on every raw frame",
        "",
        f"{len(frames)} frames: brick found in {found}, `no_brick` in {len(frames) - found} "
        f"(see `frames.csv`, `qa_skipped.jpg`). Time per frame with a brick: median {np.median(ms):.0f} ms, "
        f"95 % {np.percentile(ms, 95):.0f} ms (brick finding + model).",
        "",
        "## Speed per OpenVINO device",
        "",
        "| device | status | model only ms | full frame ms |", "|---|---|---|---|",
        *[f"| {d} | {s} | {m} | {fl} |" for d, s, m, fl in speed],
        "",
        "## Pictures",
        "",
        "Held-out good bricks (should stay blue):", "", "![good](gallery_good.jpg)", "",
        "Synthetic defects, lowest scores first (the hardest ones):", "", "![defective](gallery_defective.jpg)", "",
        "Brick finding: `qa_crops.jpg`, `qa_masks.jpg`, `qa_skipped.jpg`. Every test image: `heatmaps/`.",
        "",
        "Caveat: test good bricks come from the same sessions as training (no separate test session yet), "
        "and the defects are synthetic. Collect a separate good session and real defective bricks for "
        "a final number.",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({**summary, "per_kind": kind_rows, "per_colour": colour_rows,
                                                  "speed": speed}, indent=2))
    print(json.dumps(summary, indent=1))
    print("per kind:", kind_rows)
    print(f"report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
