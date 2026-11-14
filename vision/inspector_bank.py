"""Inspection with a bank of models (registry from train_views.py), switchable variants.

    bank = InspectorBank("results/models/models.json", mode="per_face", device="NPU")
    r = bank.inspect_face(frame, camera="front")      # brick cut out -> face -> score -> decision
    r["face"], r["decision"], r["score"]

Modes (per camera; the gripper and the front camera always have their own models):
    single      one model for every face of the camera; face unknown ("any")
    per_face    one model per face; face = model competition: every face model scores the crop, the
                lowest anomaly score wins (the frame is "normal" for its own face's model, unknown to the
                others). If the two best are close and both high -> uncertain ("face unclear").
    classifier  face classifier (train_face_classifier.py) picks the face; the competition is the
                cross-check: if a different face model fits much better -> uncertain, not a verdict.

Brick level (rule confirmed by the judges): faces one after another, the first confidently
defective face ends the inspection; uncertain faces are re-shot; the brick is good only when every
face in `required` has been seen and was good. Faces seen are tracked because the brick's starting
orientation is unknown: missing faces are reported (the arm decides how to show them).

CLI (offline check, one frame = one face):
    python vision/inspector_bank.py --camera front --mode per_face --folder data/<session> --pattern "*_ground_color.png"
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from benchmark import Scorer
from finders import make_finder

FACES = ["top", "bottom", "side"]
DEFAULT_REQUIRED = {"top": 1, "bottom": 1, "side": 4}


class InspectorBank:
    def __init__(self, registry: Path | str, mode: str = "per_face", device: str = "AUTO",
                 face_margin: float = 0.1, cls_min_prob: float = 0.8,
                 anomaly_device: str | None = None, classifier_device: str | None = None):
        """device: default for every model; anomaly_device / classifier_device override it, e.g.
        anomaly_device="NPU", classifier_device="GPU" (Anomalib on the NPU, classifier on the iGPU,
        CPU free for the robot). Unsupported devices fall back (vision/ov_utils.py); the device each
        model really runs on is in self.devices. Every model is compiled + run once here (warm-up),
        so the first brick of the demo is not slow."""
        from ov_utils import compile_with_fallback
        self.reg = json.loads(Path(registry).read_text())["cameras"]
        self.mode, self.device = mode, device
        self.face_margin, self.cls_min_prob = face_margin, cls_min_prob
        self._finders, self._scorers, self._cls = {}, {}, {}
        self.devices = {}
        a_dev, c_dev = anomaly_device or device, classifier_device or device
        for cam, e in self.reg.items():
            bg = cv2.imread(e["background"]) if e.get("background") else None
            self._finders[cam] = make_finder(e["finder"], e["camera"], bg)
            for face, m in (e.get("per_face") or {}).items():
                sc = Scorer(Path(m["xml"]), a_dev)
                self._scorers[(cam, face)] = (sc, m)
                self.devices[f"{cam}/{face}"] = sc.device_used
            if e.get("single"):
                sc = Scorer(Path(e["single"]["xml"]), a_dev)
                self._scorers[(cam, "any")] = (sc, e["single"])
                self.devices[f"{cam}/all"] = sc.device_used
            if e.get("classifier"):
                c = e["classifier"]
                r = compile_with_fallback(c["xml"], c_dev)
                if r.compiled is not None:
                    self._cls[cam] = (r.compiled, c)
                    self.devices[f"{cam}/classifier"] = r.device
        missing = {"single": "single model", "per_face": "per-face models", "classifier": "classifier"}
        for cam in self.reg:
            have = {"single": (cam, "any") in self._scorers,
                    "per_face": any(k[0] == cam and k[1] != "any" for k in self._scorers),
                    "classifier": cam in self._cls and any(k[0] == cam and k[1] != "any" for k in self._scorers)}
            if not have[mode]:
                print(f"[bank] camera {cam}: no {missing[mode]} for mode '{mode}'")

    # ------------------------------------------------------------------ helpers
    def _score(self, cam: str, face: str, crop: np.ndarray) -> float:
        sc, _ = self._scorers[(cam, face)]
        return float(np.asarray(sc(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).pred_score).ravel()[0])

    def _classify(self, cam: str, crop: np.ndarray) -> tuple[str, float]:
        model, c = self._cls[cam]
        s = c["size"]
        small = cv2.resize(crop, (s, s), interpolation=cv2.INTER_AREA)
        if c.get("ppp"):  # preprocessing is inside the model: feed the raw uint8 BGR crop
            logits = model(small[None])[0][0]
        else:
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            x = ((rgb.astype(np.float32) / 255 - np.array(c["mean"], np.float32)) / np.array(c["std"], np.float32))
            logits = model(x.transpose(2, 0, 1)[None])[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        i = int(np.argmax(p))
        return c["classes"][i], float(p[i])

    @staticmethod
    def _decide(score: float, m: dict) -> str:
        return "defective" if score >= m["high"] else "good" if score < m["low"] else "uncertain"

    # ------------------------------------------------------------------ one face
    def inspect_face(self, frame: np.ndarray, camera: str) -> dict:
        t0 = time.perf_counter()
        out = {"camera": camera, "mode": self.mode, "face": None, "decision": "uncertain", "score": None,
               "scores": {}, "reason": "", "ms": {}}
        r = self._finders[camera](frame)
        out["ms"]["crop"] = round(1000 * (time.perf_counter() - t0), 1)
        if r.crop is None or not r.ok:
            out["reason"] = "brick not found / bad crop: " + ", ".join(r.reasons or ["no brick"])
            return out
        crop = r.crop
        t1 = time.perf_counter()
        faces = [f for f in FACES if (camera, f) in self._scorers]

        if self.mode == "single":
            s = self._score(camera, "any", crop)
            m = self._scorers[(camera, "any")][1]
            out.update(face="any", score=round(s, 4), decision=self._decide(s, m),
                       reason=f"single model, low {m['low']:.2f} high {m['high']:.2f}")
        else:
            scores = {f: self._score(camera, f, crop) for f in faces}
            out["scores"] = {f: round(v, 4) for f, v in scores.items()}
            ranked = sorted(scores, key=scores.get)
            best = ranked[0]
            face = best
            if self.mode == "classifier" and camera in self._cls:
                cf, prob = self._classify(camera, crop)
                out["classifier"] = {"face": cf, "prob": round(prob, 3)}
                if prob >= self.cls_min_prob and cf in scores:
                    face = cf
                    # cross-check: another face model fits clearly better -> do not trust a verdict
                    if best != cf and scores[cf] - scores[best] > self.face_margin and scores[cf] >= \
                            self._scorers[(camera, cf)][1]["low"]:
                        out.update(face=cf, score=round(scores[cf], 4),
                                   reason=f"classifier says {cf} ({prob:.2f}) but {best} model fits better")
                        out["ms"]["model"] = round(1000 * (time.perf_counter() - t1), 1)
                        return out
            elif len(ranked) > 1:
                second = ranked[1]
                m_best = self._scorers[(camera, best)][1]
                if scores[second] - scores[best] < self.face_margin and scores[best] >= m_best["low"]:
                    out.update(face=best, score=round(scores[best], 4),
                               reason=f"face unclear ({best} {scores[best]:.2f} vs {second} {scores[second]:.2f})")
                    out["ms"]["model"] = round(1000 * (time.perf_counter() - t1), 1)
                    return out
            m = self._scorers[(camera, face)][1]
            s = scores[face]
            out.update(face=face, score=round(s, 4), decision=self._decide(s, m),
                       reason=f"{face} model, low {m['low']:.2f} high {m['high']:.2f}")
        out["ms"]["model"] = round(1000 * (time.perf_counter() - t1), 1)
        out["ms"]["total"] = round(1000 * (time.perf_counter() - t0), 1)
        out["_crop"] = crop
        return out

    # ------------------------------------------------------------------ one brick
    def inspect_brick(self, get_frame, steps: list[tuple[str, str]], retries: int = 1,
                      required: dict | None = None, brick_id=None) -> dict:
        """get_frame(step, attempt) -> BGR frame; steps = [(step_name, camera), ...] in arm order."""
        required = DEFAULT_REQUIRED if required is None else required
        t0 = time.perf_counter()
        seen, log, decision, worst = {}, [], "good", None
        for step, cam in steps:
            for attempt in range(retries + 1):
                r = self.inspect_face(get_frame(step, attempt), cam)
                log.append({k: v for k, v in r.items() if not k.startswith("_")} | {"step": step, "attempt": attempt})
                if r["decision"] != "uncertain":
                    break
            if r["decision"] == "defective":
                decision, worst = "defective", step
                break  # early exit
            if r["decision"] == "uncertain":
                decision = "uncertain"
                continue
            if r["face"]:
                seen[r["face"]] = seen.get(r["face"], 0) + 1
        missing = {f: n - seen.get(f, 0) for f, n in required.items()
                   if self.mode != "single" and seen.get(f, 0) < n}
        if decision == "good" and missing:
            decision = "incomplete"
        return {"brick": brick_id, "decision": decision, "worst_step": worst, "faces_seen": seen,
                "missing": missing if decision != "defective" else {}, "steps_checked": len({e["step"] for e in log}),
                "of": len(steps), "early_exit": decision == "defective" and len({e["step"] for e in log}) < len(steps),
                "mode": self.mode, "ms_total": round(1000 * (time.perf_counter() - t0), 1), "log": log}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registry", type=Path, default=Path("results/models/models.json"))
    p.add_argument("--mode", choices=["single", "per_face", "classifier"], default="per_face")
    p.add_argument("--camera", required=True)
    p.add_argument("--folder", type=Path, required=True)
    p.add_argument("--pattern", default="*.png")
    p.add_argument("--device", default="CPU")
    a = p.parse_args()
    bank = InspectorBank(a.registry, a.mode, a.device)
    counts = {}
    for f in sorted(a.folder.rglob(a.pattern)):
        r = bank.inspect_face(cv2.imread(str(f)), a.camera)
        counts[r["decision"]] = counts.get(r["decision"], 0) + 1
        print(f"{f.parent.name[-18:]}/{f.name:28s} face {str(r['face']):6s} {r['decision']:10s} "
              f"score {r['score'] if r['score'] is not None else '-':>6}  {r['reason']}")
    print(counts)


if __name__ == "__main__":
    main()
