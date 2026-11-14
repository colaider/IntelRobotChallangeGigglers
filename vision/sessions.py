"""Read raw collection sessions (data/<session>/, written by collection/run.py) + face labels.

Every capture of a session has one colour image per camera (ground / top / gripper), aligned
depth for the RealSense cameras and the arm pose. This module turns them into a flat list of
captures that the experiment runner (vision/experiment.py) can filter and split.

    from sessions import load_captures
    caps = load_captures()                       # all sessions in data/
    caps[0].image("ground"), caps[0].depth("top"), caps[0].face, caps[0].pose_group

Face labels live in dataset/faces.csv (session,id,face), one row per capture:
    top       studs facing the ground camera
    bottom    cavity (tube) facing the camera
    side      a side face (studs on the outline)
    none      no brick in the ground camera's view
    occluded  fingers on the brick or the brick cut by the frame border

Contact sheets for labelling (numbered tiles, one sheet per session and camera):
    python vision/sessions.py --sheet ground gripper top
"""

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

DATA = Path("data")
FACES_CSV = Path("dataset/faces.csv")
FACES = ["top", "bottom", "side", "none", "occluded"]
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def colour_of(session: Path) -> str:
    """20261114_121312_light_purple_good_a -> light_purple"""
    name = "_".join(session.name.split("_")[2:])
    for tag in ("_good", "_defective", "_test"):
        name = name.split(tag)[0]
    return name or "unknown"


@dataclass
class Capture:
    session: Path
    id: str
    colour: str
    label: str
    defect: str
    pose: dict = field(default_factory=dict)  # joint -> degrees
    face: str = ""
    depth_scale: dict = field(default_factory=dict)  # camera -> metres per unit

    @property
    def key(self) -> str:
        return f"{self.session.name}_{self.id}"

    @property
    def pose_group(self) -> str:
        """Same arm pose (2 deg bins) -> same group: near-duplicate frames stay in one split."""
        return self.session.name + ":" + ",".join(str(round(self.pose.get(j, 0) / 2)) for j in JOINTS)

    def path(self, camera: str, kind: str = "color") -> Path:
        return self.session / f"{self.id}_{camera}_{kind}.png"

    def image(self, camera: str) -> np.ndarray | None:
        return cv2.imread(str(self.path(camera)))

    def depth(self, camera: str) -> np.ndarray | None:
        """Depth in millimetres (float32, 0 = no data), aligned to the colour image."""
        d = cv2.imread(str(self.path(camera, "depth")), cv2.IMREAD_UNCHANGED)
        if d is None:
            return None
        return d.astype(np.float32) * self.depth_scale.get(camera, 0.001) * 1000


def load_faces(path: Path = FACES_CSV) -> dict[tuple[str, str], str]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {(r["session"], r["id"].zfill(4)): r["face"] for r in csv.DictReader(f)}


def load_captures(data: Path = DATA, faces_csv: Path = FACES_CSV) -> list[Capture]:
    faces = load_faces(faces_csv)
    caps = []
    for s in sorted(p for p in Path(data).iterdir() if (p / "manifest.csv").exists()):
        info = json.loads((s / "session.json").read_text())
        scales = {cam: c.get("depth_scale", 0.001) for cam, c in info.get("cameras", {}).items()}
        with (s / "manifest.csv").open(newline="") as f:
            for row in csv.DictReader(f):
                pose = {j: float(row[f"{j}_deg"]) for j in JOINTS if row.get(f"{j}_deg") not in (None, "")}
                caps.append(Capture(s, row["id"], row.get("color") or colour_of(s), row.get("label") or "good",
                                    row.get("defect", ""), pose, faces.get((s.name, row["id"]), ""), scales))
    return caps


def contact_sheet(caps: list[Capture], camera: str, cols: int = 5, w: int = 256) -> np.ndarray:
    tiles = []
    for c in caps:
        img = c.image(camera)
        h = int(w * 3 / 4)
        t = np.zeros((h, w, 3), np.uint8) if img is None else cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        text = f"{c.id} {c.face}".strip()
        cv2.putText(t, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(t, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--sheet", nargs="+", default=["ground"], metavar="CAMERA")
    p.add_argument("--out", type=Path, default=Path("results/sessions/sheets"))
    args = p.parse_args()
    caps = load_captures(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    by_session: dict[str, list[Capture]] = {}
    for c in caps:
        by_session.setdefault(c.session.name, []).append(c)
    for cam in args.sheet:
        for name, cs in by_session.items():
            out = args.out / f"{cam}_{colour_of(Path(name))}.jpg"
            cv2.imwrite(str(out), contact_sheet(cs, cam))
            print(out)
    labelled = sum(bool(c.face) for c in caps)
    print(f"{len(caps)} captures in {len(by_session)} sessions, {labelled} with a face label")


if __name__ == "__main__":
    main()
