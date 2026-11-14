"""Use the trained model: one frame in -> good / defective / no brick, score, heatmap.

In code (robot runtime needs only openvino + opencv + numpy, no Anomalib / torch):

    from inspector import Inspector
    insp = Inspector()                       # loads anomaly/model/ (CPU; device="GPU" or "NPU" too)
    r = insp.inspect(frame_bgr)              # a 640x480 ground-camera frame (BGR, as OpenCV reads)
    r.label      -> "good" | "defective" | "no_brick"
    r.score      -> 0..1 (>= threshold -> defective), None if no brick
    r.view       -> picture for humans: frame + outline | crop | heatmap, with the verdict

From the command line (image file or folder):

    .venv\\Scripts\\python.exe anomaly/inspector.py data/20261114_120510_red_good_a/0003_ground_color.png
    .venv\\Scripts\\python.exe anomaly/inspector.py data/<session> --out anomaly/reports/check

Robustness:
  * no brick in view (empty gripper, brick out of frame) -> label "no_brick", never a crash
  * wrong frame size (other camera / resolution) -> clear error message
  * every result carries the time it took; the threshold lives in model/model_info.json
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from brick import BrickFinder, overlay  # noqa: E402


@dataclass
class Result:
    label: str                              # "good" | "defective" | "no_brick"
    score: float | None = None              # normalised anomaly score, None without brick
    threshold: float = 0.5
    ms: float = 0.0                         # total time: brick finding + model
    crop: np.ndarray | None = None          # the square brick crop that was scored (BGR)
    anomaly_map: np.ndarray | None = None   # HxW float 0..1 on the crop
    mask: np.ndarray | None = None          # brick mask on the frame
    view: np.ndarray | None = field(default=None, repr=False)  # human-readable picture

    @property
    def defective(self) -> bool | None:
        return None if self.label == "no_brick" else self.label == "defective"


def heat_overlay(crop: np.ndarray, amap: np.ndarray) -> np.ndarray:
    """Heatmap on a FIXED 0..1 scale (0.5 = threshold), so good bricks stay blue."""
    a = cv2.resize(np.clip(amap, 0, 1).astype(np.float32), (crop.shape[1], crop.shape[0]))
    heat = cv2.applyColorMap((255 * a).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(crop, 0.55, heat, 0.45, 0)


class Inspector:
    def __init__(self, model_dir: Path = C.MODEL, device: str = "CPU", threshold: float | None = None):
        import openvino as ov

        model_dir = Path(model_dir)
        info_path = model_dir / "model_info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"No model in {model_dir} - run anomaly/train.py first.")
        self.info = json.loads(info_path.read_text())
        self.threshold = float(threshold if threshold is not None else self.info.get("threshold", 0.5))
        self.size = int(self.info.get("image_size", 256))
        self.finder = BrickFinder.load(model_dir / self.info.get("background", "background.png"))
        core = ov.Core()
        if not any(d.startswith(device) for d in core.available_devices):
            raise RuntimeError(f"OpenVINO device {device} not available; have {core.available_devices}")
        self.device = device
        self._model = core.compile_model(model_dir / self.info.get("openvino_xml", "openvino/model.xml"), device)
        self._outputs = {o.any_name: o for o in self._model.outputs}
        self.score_crop(np.full((self.size, self.size, 3), 128, np.uint8))  # warm-up (first call compiles)

    def score_crop(self, crop_bgr: np.ndarray) -> tuple[float, np.ndarray]:
        """Score an already cropped brick: (normalised score, anomaly map size x size)."""
        rgb = cv2.cvtColor(cv2.resize(crop_bgr, (self.size, self.size), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        out = self._model(x)
        score = float(np.asarray(out[self._outputs["pred_score"]]).ravel()[0])
        amap = np.asarray(out[self._outputs["anomaly_map"]]).reshape(self.size, self.size)
        return score, amap

    def inspect(self, frame_bgr: np.ndarray, make_view: bool = True) -> Result:
        """Full pipeline on one ground-camera frame."""
        t0 = time.perf_counter()
        mask = self.finder.mask(frame_bgr)
        if mask is None:
            r = Result("no_brick", threshold=self.threshold, ms=(time.perf_counter() - t0) * 1000)
            if make_view:
                r.view = self._view(frame_bgr, r)
            return r
        from brick import canonical_crop

        crop, _ = canonical_crop(frame_bgr, mask, self.finder.margin)
        score, amap = self.score_crop(crop)
        r = Result("defective" if score >= self.threshold else "good", score, self.threshold,
                   (time.perf_counter() - t0) * 1000, crop, amap, mask)
        if make_view:
            r.view = self._view(frame_bgr, r)
        return r

    @staticmethod
    def _view(frame: np.ndarray, r: Result, h: int = 300) -> np.ndarray:
        left = overlay(frame, r.mask)
        left = cv2.resize(left, (int(h * left.shape[1] / left.shape[0]), h))
        if r.crop is not None:
            crop = cv2.resize(r.crop, (h, h))
            heat = heat_overlay(crop, r.anomaly_map)
            img = np.hstack([left, crop, heat])
        else:
            img = left
        colour = {"good": (0, 200, 0), "defective": (0, 0, 255), "no_brick": (0, 200, 255)}[r.label]
        text = r.label.upper() + (f"  score {r.score:.2f} (thr {r.threshold:.2f})" if r.score is not None else "")
        bar = np.zeros((40, img.shape[1], 3), np.uint8)
        cv2.putText(bar, text + f"  {r.ms:.0f} ms", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
        return np.vstack([img, bar])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="image files or folders (ground-camera frames)")
    ap.add_argument("--out", type=Path, default=C.REPORTS / "inspect", help="where to save the pictures + CSV")
    ap.add_argument("--device", default="CPU", help="CPU | GPU | NPU")
    ap.add_argument("--threshold", type=float, default=None, help="override model_info.json")
    ap.add_argument("--all-cameras", action="store_true", help="in folders, also take non-ground images")
    args = ap.parse_args()

    files = []
    for p in args.inputs:
        if p.is_dir():
            pat = "*.png" if args.all_cameras else f"*_{C.CAMERA}_color.png"
            files += sorted(p.glob(pat)) or sorted(p.glob("*.png"))
        else:
            files.append(p)
    insp = Inspector(device=args.device, threshold=args.threshold)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{'image':50s} {'label':10s} {'score':>6s} {'ms':>6s}")
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"{str(f)[-50:]:50s} cannot read")
            continue
        r = insp.inspect(img)
        name = f"{f.parent.name}_{f.stem}"
        cv2.imwrite(str(args.out / f"{name}.jpg"), r.view)
        rows.append({"image": str(f), "label": r.label, "score": "" if r.score is None else round(r.score, 4),
                     "ms": round(r.ms, 1)})
        print(f"{str(f)[-50:]:50s} {r.label:10s} {'' if r.score is None else f'{r.score:.3f}':>6s} {r.ms:6.0f}")
    with (args.out / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, ["image", "label", "score", "ms"])
        w.writeheader()
        w.writerows(rows)
    print(f"\npictures + results.csv in {args.out}")


if __name__ == "__main__":
    main()
