"""Inspection decision: camera frame(s) of one brick -> good / defective / uncertain (+ JSON for the robot).

    face   = one camera frame showing one face of the brick
    brick  = faces checked one after another, with EARLY EXIT (rule confirmed by the judges):
             the first confidently defective face ends the inspection -> defective bowl.
             An uncertain face is re-shot (up to --retries) before deciding.
             A brick is good only when every face is confidently good.

Per face: segment.process (find + straighten + crop the brick; a bad mask -> uncertain)
          -> OpenVINO PatchCore -> score on the brick -> two thresholds:
             score <  low   -> good
             score >= high  -> defective
             in between     -> uncertain
Thresholds come from vision/calibrate.py (JSON); defaults low 0.4 / high 0.6 around Anomalib's 0.5.

"Score on the brick": the model score scaled by how much of the heat-map peak lies ON the brick
(eroded crop mask). A hot spot on a leftover bit of gripper next to the brick no longer raises
the score; a marker on the brick keeps it.

Robot code:
    insp = Inspector("model.xml", device="NPU", calib="results/calibration.json")
    result = insp.inspect_brick(lambda face, attempt: grab_frame_at_pose(face), faces=["A", "B", "top"])
    result["decision"]  # "good" | "defective" | "uncertain"

CLI (offline check; images are the faces of ONE brick, in inspection order):
    python vision/inspect_brick.py --xml results/export/patchcore/fp16/weights/openvino/model.xml \
        --calib results/calibration.json --images f1.png f2.png f3.png
    python vision/inspect_brick.py --xml ... --folder dataset/raw/test/defective   # one face per brick
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from anomalib.deploy import OpenVINOInferencer

from benchmark import input_hw
from segment import IMAGE_EXT, process

DEFAULT_LOW, DEFAULT_HIGH = 0.4, 0.6


class Inspector:
    def __init__(self, xml: Path, device: str = "AUTO", seg: str | None = "scene", window=None,
                 calib: Path | None = None, low: float | None = None, high: float | None = None,
                 erode_frac: float = 0.02):
        self.inf = OpenVINOInferencer(path=xml, device=device)
        self.hw = input_hw(Path(xml)) or (256, 256)
        self.seg, self.window, self.erode_frac = seg, window, erode_frac
        cal = json.loads(Path(calib).read_text()) if calib else {}
        self.low = low if low is not None else cal.get("low", DEFAULT_LOW)
        self.high = high if high is not None else cal.get("high", DEFAULT_HIGH)
        self.device = device

    # ------------------------------------------------------------------ one face
    def inspect_face(self, bgr: np.ndarray, face: str = "face") -> dict:
        t0 = time.perf_counter()
        out = {"face": face, "decision": "uncertain", "score": None, "reason": "", "ms": {}}
        if self.seg:
            try:
                _, crop, crop_m, info = process(bgr, mode=self.seg, window=self.window)
            except ValueError:
                out["reason"] = "no brick found"
                out["ms"]["segment"] = round(1000 * (time.perf_counter() - t0), 1)
                return out
            out["ms"]["segment"] = round(1000 * (time.perf_counter() - t0), 1)
            if not info["ok"]:
                out["reason"] = "segmentation doubtful: " + ", ".join(info["reasons"])
                return out
        else:
            crop, crop_m = bgr, np.ones(bgr.shape[:2], np.uint8)

        t1 = time.perf_counter()
        h, w = self.hw
        rgb = cv2.cvtColor(cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        r = self.inf.predict(rgb)
        out["ms"]["model"] = round(1000 * (time.perf_counter() - t1), 1)

        model_score = float(np.asarray(r.pred_score).ravel()[0])
        amap = np.asarray(r.anomaly_map, np.float32).squeeze()
        m = cv2.resize(crop_m, (amap.shape[1], amap.shape[0]), interpolation=cv2.INTER_NEAREST)
        k = max(1, int(self.erode_frac * amap.shape[0])) * 2 + 1
        m = cv2.erode(m, np.ones((k, k), np.uint8))
        peak_all = float(amap.max())
        peak_brick = float(amap[m > 0].max()) if m.any() else peak_all
        score = model_score * (peak_brick / peak_all if peak_all > 1e-6 else 1.0)
        y, x = np.unravel_index(np.argmax(np.where(m > 0, amap, -1)), amap.shape)

        out.update(score=round(score, 4), model_score=round(model_score, 4),
                   hotspot=[round(x / amap.shape[1], 3), round(y / amap.shape[0], 3)])
        if score >= self.high:
            out["decision"], out["reason"] = "defective", f"score {score:.2f} >= {self.high:.2f}"
        elif score < self.low:
            out["decision"], out["reason"] = "good", f"score {score:.2f} < {self.low:.2f}"
        else:
            out["reason"] = f"score {score:.2f} between {self.low:.2f} and {self.high:.2f}"
        out["ms"]["total"] = round(1000 * (time.perf_counter() - t0), 1)
        out["_heatmap"], out["_crop"] = amap, crop  # for dashboards; dropped from JSON
        return out

    # ------------------------------------------------------------------ one brick
    def inspect_brick(self, get_frame, faces: list[str], retries: int = 1, brick_id=None,
                      uncertain_as: str = "uncertain") -> dict:
        """get_frame(face, attempt) -> BGR frame of that face (attempt 0 = first shot).

        Early exit on the first defective face; an uncertain face is re-shot up to `retries` times.
        If a face stays uncertain the brick is `uncertain_as` ("uncertain" -> e.g. a third bowl or a
        regrasp; "defective" -> conservative quality policy).
        """
        t0 = time.perf_counter()
        checked, scores, log = [], {}, []
        decision, worst = "good", None
        for face in faces:
            for attempt in range(retries + 1):
                r = self.inspect_face(get_frame(face, attempt), face)
                log.append({k: v for k, v in r.items() if not k.startswith("_")} | {"attempt": attempt})
                if r["decision"] != "uncertain":
                    break
            checked.append(face)
            scores[face] = r["score"]
            prev = scores.get(worst) if worst is not None else None
            if r["score"] is not None and (prev is None or r["score"] > prev):
                worst = face
            if r["decision"] == "defective":
                decision, worst = "defective", face
                break  # early exit: no need to look at the remaining faces
            if r["decision"] == "uncertain":
                decision = uncertain_as
                if uncertain_as == "defective":
                    worst = face
                    break
        valid = [s for s in scores.values() if s is not None]
        return {
            "brick": brick_id,
            "decision": decision,
            "faces": scores,
            "max_score": max(valid) if valid else None,
            "worst_face": worst,
            "checked": len(checked),
            "of": len(faces),
            "early_exit": decision == "defective" and len(checked) < len(faces),
            "thresholds": {"low": self.low, "high": self.high},
            "device": self.device,
            "ms_total": round(1000 * (time.perf_counter() - t0), 1),
            "log": log,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xml", type=Path, required=True, help="exported OpenVINO model.xml (trained with --seg)")
    p.add_argument("--calib", type=Path, default=None, help="thresholds JSON from calibrate.py")
    p.add_argument("--device", default="CPU")
    p.add_argument("--seg", choices=["scene", "paper", "none"], default="scene")
    p.add_argument("--images", type=Path, nargs="+", help="faces of ONE brick, in inspection order")
    p.add_argument("--folder", type=Path, help="every image = a one-face brick (quick check)")
    p.add_argument("--retries", type=int, default=0, help="re-shots of an uncertain face (offline: same image)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    insp = Inspector(args.xml, args.device, None if args.seg == "none" else args.seg, calib=args.calib)
    if args.images:
        frames = {f"face{i + 1}": cv2.imread(str(p)) for i, p in enumerate(args.images)}
        res = insp.inspect_brick(lambda f, a: frames[f], list(frames), retries=args.retries)
        print(json.dumps(res, indent=2))
        return
    if args.folder:
        files = sorted(p for p in args.folder.rglob("*") if p.suffix.lower() in IMAGE_EXT)
        counts = {"good": 0, "defective": 0, "uncertain": 0}
        for p in files:
            r = insp.inspect_face(cv2.imread(str(p)), p.stem)
            counts[r["decision"]] += 1
            print(f"{p.name:44s} {r['decision']:10s} score {r['score'] if r['score'] is not None else '-':>6}  "
                  f"{r['reason']}  ({r['ms'].get('total', r['ms'].get('segment', 0))} ms)")
        print(f"\n{counts}  (thresholds low {insp.low:.2f}, high {insp.high:.2f})")


if __name__ == "__main__":
    main()
