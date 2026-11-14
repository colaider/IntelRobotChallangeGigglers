# Dataset layout

**Robot collection (current team format):** `data/<session>/` from `collection/run.py` (gitignored,
shared as zip / drive) + face labels in `dataset/faces.csv` (committed) → `vision/build_views.py`
→ `dataset/views/<view>/` (gitignored). Step by step: [`docs/dataset_workflow.md`](../docs/dataset_workflow.md).

**Hand-held capture (older, still works):** the layout below.

```
dataset/
  raw/                     # full camera frames from vision/capture.py (committed)
    training/good/         # only perfect bricks; used to train
    test/good/             # perfect bricks NOT in training; threshold tuning
    test/defective/        # defective bricks (e.g. marker drawings); validation only
  processed/               # centre-cropped model input from vision/preprocess.py (gitignored)
  archive/table_v1/        # first 54 centre-cropped photos (bricks on a white table)
```

Flow:

```bash
python vision/capture.py --split training --cls good      # -> dataset/raw/training/good
python vision/train.py --seg scene                        # segments dataset/raw, then trains (recommended)
# older: python vision/preprocess.py --crop 0 && python vision/train.py   (centre crop, brick must be centred)
```

Current raw set (Ivan, 14 Nov): 30 `training/good` frames of bricks held in the gripper on a busy
background; `test/` still to be shot in the same setup.

File names: `<date>_<time>_<uuid8>_<NNN>.png` (set by capture.py, unique across machines).
Do not mix the same photo into training and test. Keep camera, light and distance the same
within one dataset version; when the setup changes, move the old set under `archive/`.
