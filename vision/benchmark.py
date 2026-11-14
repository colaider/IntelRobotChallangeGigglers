"""Benchmark exported OpenVINO models on CPU / GPU / NPU: latency, throughput and accuracy.

Finds every model.xml under --models, runs it on each available device, and reports
end-to-end latency (pre-processing + inference + post-processing, as the robot sees it),
p50/p95, FPS, image AUROC/F1 on the test set, first-inference (compile) time and model size.

Example (station laptop):
    python vision/benchmark.py --models results/export --devices CPU GPU NPU
Raw-inference numbers without Python overhead: OpenVINO's own CLI
    benchmark_app -m results/export/patchcore/int8/weights/openvino/model.xml -d NPU
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
from anomalib.deploy import OpenVINOInferencer

from common import auroc, f1, list_test_set


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", type=Path, default=Path("results/export"), help="folder searched for model.xml")
    p.add_argument("--devices", nargs="+", default=["CPU", "GPU", "NPU"])
    p.add_argument("--root", type=Path, default=Path("dataset/processed"), help="dataset root with test/good, test/defective")
    p.add_argument("--runs", type=int, default=50, help="timed runs for latency")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--csv", type=Path, default=Path("results/benchmark.csv"))
    return p.parse_args()


def model_size_mb(xml: Path) -> float:
    return (xml.stat().st_size + xml.with_suffix(".bin").stat().st_size) / 1e6


def input_hw(xml: Path) -> tuple[int, int] | None:
    """Static (H, W) of the model input, or None if dynamic."""
    shape = ov.Core().read_model(xml).inputs[0].partial_shape
    if shape.is_static:
        return int(shape[2].get_length()), int(shape[3].get_length())
    return None


class Scorer:
    """OpenVINO inferencer + the resize a static-shape model needs. Use this from robot code too."""

    def __init__(self, xml: Path, device: str = "AUTO", roi: bool = False, seg: str | None = None,
                 seg_window=None, fallback: bool = True):
        """seg="scene"|"paper": find + straighten + crop the brick first (model trained with --seg).

        After each call self.last_seg holds the segmentation info; last_seg["ok"] is False when
        the mask looks wrong -> robot code should treat the brick as 'uncertain' and re-inspect.
        roi=True: older plain-background cut-out (model trained with --roi).
        """
        # compile with device fallback (NPU -> HETERO:NPU,CPU -> GPU -> CPU), model cache and the
        # latency hint (vision/ov_utils.py), then hand the compiled model to Anomalib's inferencer
        from ov_utils import compile_with_fallback
        r = compile_with_fallback(xml, device, fallback=fallback)
        if r.compiled is None:
            raise RuntimeError(f"cannot compile {xml} on {device}: {r.tried}")
        inf = OpenVINOInferencer.__new__(OpenVINOInferencer)
        inf.device, inf.config, inf.model = r.device, None, r.compiled
        inf.input_blob, inf.output_blob = r.compiled.input(0), r.compiled.output(0)
        self.inf, self.device_used, self.compile_s, self.tried = inf, r.device, r.compile_s, r.tried
        if device.upper() not in ("AUTO", r.device):
            print(f"[openvino] {Path(xml).parent.parent.parent.name or Path(xml).name}: {device} not usable -> running on {r.device}")
        self.hw = input_hw(xml)
        self.roi = roi
        self.seg, self.seg_window, self.last_seg = seg, seg_window, None

    def __call__(self, rgb: np.ndarray):
        if self.seg:
            from segment import process
            _, crop, _, self.last_seg = process(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mode=self.seg,
                                                window=self.seg_window)  # ValueError if no brick
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        elif self.roi:
            from roi import align
            crop, _ = align(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), size=self.hw[0] if self.hw else 256)
            if crop is None:
                raise ValueError("no brick found in frame")  # robot code: treat as 'uncertain'
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        if self.hw and rgb.shape[:2] != self.hw:
            rgb = cv2.resize(rgb, (self.hw[1], self.hw[0]), interpolation=cv2.INTER_AREA)
        return self.inf.predict(rgb)


def bench(xml: Path, device: str, test_set, runs: int, warmup: int) -> dict:
    t0 = time.perf_counter()
    score = Scorer(xml, device, fallback=False)  # measure exactly this device, never a silent fallback
    images = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p, _ in test_set]
    score(images[0])
    compile_s = time.perf_counter() - t0

    # accuracy: one pass over the test set
    scores, preds = [], []
    for img in images:
        r = score(img)
        scores.append(float(np.asarray(r.pred_score).ravel()[0]))
        preds.append(bool(np.asarray(r.pred_label).ravel()[0]))
    labels = [lab for _, lab in test_set]

    # latency (incl. resize): cycle through test images
    for i in range(warmup):
        score(images[i % len(images)])
    times = []
    for i in range(runs):
        t = time.perf_counter()
        score(images[i % len(images)])
        times.append((time.perf_counter() - t) * 1000)

    return {
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "fps": 1000 / float(np.mean(times)),
        "auroc": auroc(labels, scores),
        "f1": f1(labels, preds),
        "compile_s": compile_s,
    }


def main() -> None:
    args = parse_args()
    available = ov.Core().available_devices
    devices = [d for d in args.devices if any(a.startswith(d) for a in available)]
    skipped = [d for d in args.devices if d not in devices]
    print(f"OpenVINO devices: {available}" + (f"; skipping {skipped}" if skipped else ""))

    test_set = list_test_set(args.root)
    if not test_set:
        raise SystemExit(f"No test images under {args.root}/test")
    xmls = sorted(args.models.rglob("model.xml"))
    if not xmls:
        raise SystemExit(f"No model.xml under {args.models} (run vision/export.py first)")

    rows = []
    for xml in xmls:
        name = "/".join(xml.relative_to(args.models).parts[:2])  # e.g. patchcore/int8
        for dev in devices:
            try:
                r = bench(xml, dev, test_set, args.runs, args.warmup)
            except Exception as e:  # e.g. op not supported on NPU
                print(f"{name:24s} {dev:4s} FAILED: {str(e).splitlines()[0][:120]}")
                continue
            row = {"model": name, "device": dev, "size_mb": model_size_mb(xml), **r}
            rows.append(row)
            print(f"{name:24s} {dev:4s} p50 {r['p50_ms']:7.1f} ms  p95 {r['p95_ms']:7.1f} ms  "
                  f"{r['fps']:6.1f} fps  AUROC {r['auroc']:.3f}  F1 {r['f1']:.3f}  "
                  f"compile {r['compile_s']:.1f}s  {row['size_mb']:.0f} MB")

    if rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved {args.csv}")


if __name__ == "__main__":
    main()
