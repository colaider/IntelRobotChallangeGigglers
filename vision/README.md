# Vision: anomaly detection for the bricks

## Quick start (capture)

Run in **Terminal.app** (not via `!` in Claude Code): the preview window needs the keyboard
for SPACE / Q, and the script must run from the repo root with the venv active.

```bash
cd ~/projects/IntelRobotChallangeGigglers && source .venv/bin/activate && python vision/capture.py --split training --cls good --crop 700
```

macOS asks for camera access for Terminal on the first run. Step-by-step guide for the
photographer: [`docs/photo-guide.md`](../docs/photo-guide.md).

## Setup (once)

Python 3.12. From the repo root:

```bash
uv venv --python 3.12 .venv      # or: python3.12 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt   # or: pip install -r requirements.txt
```

On Linux/Windows without an NVIDIA GPU, install CPU torch first so pip does not pull CUDA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 1. Capture photos

```bash
python vision/capture.py --split training --cls good --crop 700
python vision/capture.py --split test --cls good --crop 700
python vision/capture.py --split test --cls defective --crop 700
```

- The green box in the preview is what gets saved. Place the brick so it fills most of it.
  Models resize everything to 256x256, so a tight square crop matters.
- Keep the same camera, light, background, distance and `--crop` for all three folders.
- Files are named `<date>_<time>_c<crop>_<NNN>.png`, so several people can add photos without
  clashes, and a different crop shows up in the name (`ls dataset/*/* | grep -v _c700_`).
- Layout and rules: [`dataset/README.md`](../dataset/README.md).
- Aim for 20–30 in `training/good` and at least 10 each in `test/good` and `test/defective`
  (half of `test/` is used to pick the threshold).

## 2. Train

```bash
python vision/train.py                    # PatchCore on dataset/
python vision/train.py --model padim
```

Prints test metrics (image AUROC, F1) and the checkpoint path under `results/`.

## 3. Predict + heatmaps

```bash
python vision/predict.py --ckpt results/Patchcore/bricks/v0/weights/lightning/model.ckpt \
    --input dataset/test/defective --out results/predict
```

Prints score and label per image and writes `*_heat.png`. Heatmaps use a fixed scale:
score 0.5 is the decision threshold, so a good brick should stay blue.

`--accelerator cpu` is the default (MPS on macOS crashes in torch).
