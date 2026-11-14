"""All paths and settings of the anomaly pipeline in one place."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent          # anomaly/
DATA = ROOT.parent / "data"                     # sessions written by collection/run.py
WORK = ROOT / "work"                            # generated dataset, training runs (rebuildable)
DATASET = WORK / "dataset"                      # Anomalib layout: training/good, test/good, test/defective
MODEL = ROOT / "model"                          # the trained model you keep and use
REPORTS = ROOT / "reports"                      # one folder per evaluation run

CAMERA = "ground"        # which camera of each session is inspected
IMAGE_SIZE = 256         # model input (square); crops are resized to this
TEST_FRACTION = 0.10     # share of good bricks per colour held out for testing
MARKS_PER_TEST_IMAGE = 6  # synthetic defects drawn on each held-out good brick
SEED = 42


def colour_of(session_dir: Path, manifest_colour: str = "") -> str:
    """Brick colour: the manifest's colour, else the session name without date and label.
    20261114_120510_light_purple_good_a -> light_purple"""
    if manifest_colour:
        return manifest_colour
    name = "_".join(session_dir.name.split("_")[2:])
    for tag in ("_good", "_defective", "_test"):
        if tag in name:
            name = name.split(tag)[0]
    return name or "unknown"
