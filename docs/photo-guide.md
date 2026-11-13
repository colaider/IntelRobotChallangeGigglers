# Photo guide: first test (one colour, one face)

Goal: learn the tool (capture → train → heatmaps). These photos will be retaken later
with the robot, so speed matters more than perfection, but the setup must stay fixed.

## You need

Laptop with webcam (or USB webcam), white A4 sheet, 3 bricks of one colour (red or blue),
marker, desk lamp.

## Setup (do not touch after the first photo)

1. A4 sheet on the table.
2. Camera fixed above the sheet looking down, ~20 cm high (books, tripod, tape). Not hand-held.
3. Lamp from the side at ~45°, ceiling light off.
4. Brick lies on its side, a side face up. Always photograph this same face type.
5. Start capture (see command below). Adjust the camera so the brick fills most of the
   green box. Keep `--crop 700` (or whatever you pick) **the same for all three folders**.

## Command (Terminal.app, from the repo root)

```bash
cd ~/projects/IntelRobotChallangeGigglers && source .venv/bin/activate
python vision/capture.py --split training --cls good --crop 700
```

SPACE saves, Q quits. Change `--split` / `--cls` for the other folders.

## What to shoot

| Folder | Count | What |
|---|---|---|
| `training/good` | 25 | clean bricks #1/#2/#3: 9 / 8 / 8 frames |
| `test/good` | ≥10 | clean bricks, new positions (not the same frames as training) |
| `test/defective` | ≥10 | marker: line, stroke, dot, tiny dot 1–2 mm, defect near the edge — 2 frames each |

Between frames shift the brick 2–5 mm and rotate 5–10°. Do not move the camera or the lamp.
Draw the defects only after `training/good` and `test/good` are done.

## Then

```bash
python vision/train.py
python vision/predict.py --ckpt results/Patchcore/bricks/v0/weights/lightning/model.ckpt --input dataset/test --out results/predict
```

Write down what you see (does the marker light up, do highlights/shadows?) in `docs/experiments.md`.
