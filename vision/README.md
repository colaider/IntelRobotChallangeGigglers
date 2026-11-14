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

## 1b. Segment: find, straighten and crop the brick (recommended)

```bash
python vision/segment.py --root dataset --out dataset_seg
```

Finds the brick (Lab colour distance from the paper; `brick_mask_depth` is ready for the
RealSense), cleans the mask with morphology (open, close, largest blob, fill holes), rotates
the brick long-side-horizontal with the studs on top, crops it tight and paints the background
flat grey. Check `dataset_seg/qa_masks.jpg` and `qa_crops.jpg` after every run. Then train with
`--root dataset_seg`. The robot must call `segment.process()` on each frame before scoring.

Background is not a plain sheet (brick in the gripper, table, cables)? Use `--mode scene`
(colour regions + GrabCut, slower, ~1 s per photo). One photo, with every stage saved:
`python vision/segment.py --image <photo> --out results/one --mode scene`.
Results: [`docs/experiments.md`](../docs/experiments.md) (AUROC 0.89 -> 1.00 on our photos).

## 2. Train

```bash
python vision/train.py                    # PatchCore on dataset/
python vision/train.py --root dataset_seg # on segmented crops (better, see 1b)
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

## 4. Synthetic defects (evaluation only)

```bash
python vision/synth_defects.py --input dataset/test/good --copy-good dataset/training/good \
    --out dataset_synth --per-image 3
python vision/train.py --root dataset_synth
```

Draws marker-like lines, curves, dots, tiny 1–2 mm dots, scribbles and edge defects on the
brick (found by colour saturation), with pixel masks in `test/masks/`. `dataset_synth/` is
gitignored. Never put synthetic images into `training/`.

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
