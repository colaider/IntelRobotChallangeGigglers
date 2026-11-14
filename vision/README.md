# Vision: anomaly detection for the bricks

## Quick start (capture)

Run in **Terminal.app** (not via `!` in Claude Code): the preview window needs the keyboard
for SPACE / Q, and the script must run from the repo root with the venv active.

```bash
cd ~/projects/IntelRobotChallangeGigglers && source .venv/bin/activate && python vision/capture.py --split training --cls good
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
python vision/capture.py --split training --cls good
python vision/capture.py --split test --cls good
python vision/capture.py --split test --cls defective
python vision/preprocess.py --crop 0        # dataset/raw -> dataset/processed (model input)
```

- `capture.py` saves the **full frame** to `dataset/raw/...`; the green box is only a preview.
  Cropping happens later (`preprocess.py --crop N` centre square, or `train.py --roi`), so
  raw frames can be re-cropped without reshooting.
- `preprocess.py` refuses to overwrite: delete `dataset/processed` (or `vision/clear_data.sh`,
  which moves raw + processed to the macOS Trash) before re-running. `--median 3|5` exists but
  made no difference in our test (frames are downscaled to 256 px anyway).
- Keep the same camera, light, background and distance within one dataset version.
- Files are named `<date>_<time>_<uuid8>_<NNN>.png`, unique across machines.
- Layout and rules: [`dataset/README.md`](../dataset/README.md).
- Aim for 20–30 in `training/good` and at least 10 each in `test/good` and `test/defective`
  (half of `test/` is used to pick the threshold).

## 1b. Segment: find, straighten and crop the brick (recommended)

```bash
python vision/train.py --seg scene                 # segments dataset/raw on the fly, then trains
python vision/predict.py --ckpt ... --input <raw frames> --seg scene
python vision/segment.py --root dataset/raw --out dataset_seg --mode scene   # just segment + QA sheets
python vision/segment.py --image <photo> --out results/one --mode scene      # every stage of one photo
```

`vision/segment.py` (Alex, refined): find the brick → clean the mask → drop glued-on bits of
hand/gripper (colour + outline trim) → rotate long side horizontal, **studs on top** → tight crop,
flat grey background. Modes: `scene` = cluttered background / brick in the gripper (mean-shift +
k-means + GrabCut, ~0.7 s per Full HD frame), `paper` = plain sheet (fast). `--window X0 Y0 X1 Y1`
limits the search to where the arm presents the brick. A quality check (area, aspect, solidity,
frame border, how much was trimmed) marks doubtful masks → treat as `uncertain` and re-inspect
(`Scorer(..., seg="scene").last_seg["ok"]`). `brick_mask_depth` is ready for the RealSense.
Always look at `qa_crops.jpg` after segmenting a new dataset.

Why: on Ivan's gripper frames the brick moves around the frame; a centre crop fails as soon as
it is off-centre (AUROC 0.58), segmentation stays at 1.00 — see `docs/experiments.md`.

## 2. Train

```bash
python vision/train.py                    # PatchCore on dataset/processed (centre crop)
python vision/train.py --seg scene        # on segmented bricks from dataset/raw (recommended)
python vision/train.py --model padim
```

Prints test metrics (image AUROC, F1) and the checkpoint path under `results/`.

## 3. Predict + heatmaps

```bash
python vision/predict.py --ckpt results/Patchcore/bricks/v0/weights/lightning/model.ckpt \
    --input dataset/processed/test/defective --out results/predict
```

Prints score and label per image and writes `*_heat.png`. Heatmaps use a fixed scale:
score 0.5 is the decision threshold, so a good brick should stay blue.

`--accelerator cpu` is the default (MPS on macOS crashes in torch).

## 4. Synthetic defects (evaluation only)

```bash
python vision/synth_defects.py --input dataset/processed/test/good --copy-good dataset/processed/training/good \
    --out dataset_synth --per-image 3
python vision/train.py --root dataset_synth
```

Draws marker-like lines, curves, dots, tiny 1–2 mm dots, scribbles and edge defects on the
brick (found by colour saturation), with pixel masks in `test/masks/`. `dataset_synth/` is
gitignored. Never put synthetic images into `training/`.

Inks: `black blue red green purple orange yellow tone` (`tone` = darker shade of the brick's
own colour, the hardest case).

**Sensitivity table** — smallest defect we catch, per marker colour and brick colour:

```bash
python vision/synth_defects.py --input dataset/processed/test/good --out dataset_sweep \
    --sweep 4 8 12 16 24 32 --inks black blue red purple tone
python vision/sensitivity.py --xml results/export/patchcore/fp16/weights/openvino/model.xml \
    --root dataset_sweep --by-colour
```

Brick colour comes from an HSV check (`synth_defects.brick_colour`), the same one planned for
routing bricks to per-colour models. It only picks a colour, never judges defects.

## 4b. Augment good images (optional)

```bash
python vision/train.py --augment 3            # + 3 variations per good image
python vision/train.py --augment 3 --aug-flip # also mirror (symmetric faces only)
```

Small rotation/shift/scale and brightness/contrast only — never colour changes, which would
make a coloured marker look "normal". `--seed` (default 42) makes runs comparable.

## 4c. Brick ROI (older, plain background only — prefer `--seg`)

```bash
python vision/train.py --roi                       # train/test on aligned brick-only crops
python vision/predict.py --ckpt ... --input ... --roi
```

`vision/roi.py` finds the brick per frame (chroma/saturation vs the border colour), straightens
it, crops it and greys out gripper/table, so only the brick is judged. Off by default; use the
same `--roi` for training and inference (`benchmark.Scorer(..., roi=True)`).

## 5. Export to OpenVINO (FP32 / FP16 / INT8)

```bash
python vision/export.py --ckpt results/Patchcore/bricks/v0/weights/lightning/model.ckpt
```

Writes `results/export/<model>/{fp32,fp16,int8}/weights/openvino/model.xml`. INT8 = NNCF
post-training quantisation calibrated on `training/good`. Models are exported with a static
input `[1, 3, 256, 256]` (Anomalib 2.7's dynamic export bakes in batch=2 for PatchCore; static
shapes are also what the NPU needs), so frames must be resized to 256×256 first — use
`benchmark.Scorer`, which does it.

## 6. Benchmark CPU / GPU / NPU

```bash
python vision/benchmark.py --models results/export --devices CPU GPU NPU
benchmark_app -m results/export/patchcore/int8/weights/openvino/model.xml -d NPU   # raw inference
```

Per model × device: p50/p95 latency (incl. resize + pre/post-processing), FPS, AUROC, F1,
compile time, size → `results/benchmark.csv`. Devices missing on the machine are skipped
(a Mac only has CPU).

## 7. Compare models

```bash
python vision/compare.py --models patchcore padim stfpm --precision int8 --devices CPU GPU NPU
```

Trains, exports and benchmarks each model → table + `results/compare/compare.csv`.
`efficient_ad` (downloads ImageNette ~1.5 GB) and `dinomaly` (heavy) are opt-in.
