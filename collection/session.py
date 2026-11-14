"""Write one collection session to disk: raw images, aligned depth, robot pose, metadata.

Layout (one folder per sitting: same light, same setup):

    data/<YYYYmmdd_HHMMSS>_<name>/
      session.json            cameras (model, serial, intrinsics, depth scale), robot port, notes,
                              git commit of this code
      manifest.csv            one row per capture (label, defect, colour, quality, joint degrees, files)
      0001_realsense_color.png    raw colour (PNG, lossless)
      0001_realsense_depth.png    uint16 depth aligned to colour (raw units; x depth_scale = metres)
      0001_cam1_color.png         extra USB cameras, if any
      0001_meta.json              everything about this capture (pose ticks+deg, exposure, timestamps)

Nothing is cropped or enhanced here: post-processing (crop, CLAHE, export to Anomalib folders)
lives in post.py and always starts again from these raw files.
"""

import csv
import json
import subprocess
import time
from pathlib import Path

import cv2

from cameras import Frame, quality
from robot import JOINTS

MANIFEST_FIELDS = (["id", "time", "label", "defect", "color", "note", "sharpness", "overexposed_pct",
                    "brightness", "auto_exposure"] + [f"{j}_deg" for j in JOINTS.values()] + ["files"])


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


class SessionWriter:
    def __init__(self, root: Path, name: str = "", notes: str = "", cameras: dict | None = None,
                 robot: dict | None = None):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.dir = Path(root) / (f"{stamp}_{name}" if name else stamp)
        self.dir.mkdir(parents=True, exist_ok=False)
        self.manifest = self.dir / "manifest.csv"
        self.count = 0
        self._last: list[Path] = []
        info = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "name": name, "notes": notes,
                "code_commit": git_commit(), "cameras": cameras or {}, "robot": robot or {},
                "joint_names": JOINTS, "depth_format": "uint16 PNG, aligned to colour, metres = value * depth_scale"}
        (self.dir / "session.json").write_text(json.dumps(info, indent=2))
        with self.manifest.open("w", newline="") as f:
            csv.DictWriter(f, MANIFEST_FIELDS).writeheader()

    def save(self, frames: dict[str, Frame], pose: dict | None, label: str, defect: str = "",
             color: str = "", note: str = "") -> dict:
        """Write one capture. frames: {camera_name: Frame}. Returns the manifest row."""
        self.count += 1
        cid = f"{self.count:04d}"
        files, meta_frames = [], {}
        for cam, fr in frames.items():
            p = self.dir / f"{cid}_{cam}_color.png"
            cv2.imwrite(str(p), fr.color)
            files.append(p)
            if fr.depth is not None:
                pd = self.dir / f"{cid}_{cam}_depth.png"
                cv2.imwrite(str(pd), fr.depth)  # uint16 PNG keeps every depth value exactly
                files.append(pd)
            meta_frames[cam] = {**fr.meta, "quality": quality(fr.color)}
        main = next(iter(meta_frames.values()))
        meta = {"id": cid, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "label": label, "defect": defect,
                "color": color, "note": note, "pose": pose, "frames": meta_frames,
                "files": [p.name for p in files]}
        mp = self.dir / f"{cid}_meta.json"
        mp.write_text(json.dumps(meta, indent=2))
        files.append(mp)
        row = {"id": cid, "time": meta["time"], "label": label, "defect": defect, "color": color, "note": note,
               **{k: main["quality"][k] for k in ("sharpness", "overexposed_pct", "brightness")},
               "auto_exposure": main.get("auto_exposure", ""),
               **{f"{j}_deg": (pose or {}).get(j, {}).get("deg", "") for j in JOINTS.values()},
               "files": " ".join(p.name for p in files)}
        with self.manifest.open("a", newline="") as f:
            csv.DictWriter(f, MANIFEST_FIELDS).writerow(row)
        self._last = files
        return row

    def delete_last(self) -> str | None:
        """Undo the last capture (files + manifest row). Only one step back."""
        if not self._last:
            return None
        cid = self._last[0].name[:4]
        for p in self._last:
            p.unlink(missing_ok=True)
        with self.manifest.open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["id"] != cid]
        with self.manifest.open("w", newline="") as f:
            w = csv.DictWriter(f, MANIFEST_FIELDS)
            w.writeheader()
            w.writerows(rows)
        self._last = []
        self.count -= 1
        return cid
