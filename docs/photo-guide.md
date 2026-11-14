# Photo guide

Two setups so far: **(A) brick held in the gripper** in front of the inspection camera (current,
`dataset/raw`, closest to the robot) and **(B) brick on a white sheet** (first test, archived in
`dataset/archive/table_v1`). The rules below apply to both; setup-specific notes are marked.

For (A): keep the camera fixed, present the brick at roughly the same distance, keep it in focus
(blurred close-ups lose the studs), and if possible put a plain dark/grey board behind the
inspection spot (people, lamps and a pale gripper next to a pale brick are the main error sources).

The steps below describe (B), the original first test.

## You need

Laptop with webcam (or USB webcam), white A4 sheet, 3 bricks of one colour (red or blue),
marker, desk lamp.

## Setup (do not touch after the first photo)

1. A4 sheet on the table.
2. Camera fixed above the sheet looking down, ~20 cm high (books, tripod, tape). Not hand-held.
3. Lamp from the side at ~45°, ceiling light off.
4. Brick lies on its side, a side face up. Always photograph this same face type.
5. Start capture (see command below). Full frames are saved to `dataset/raw/`; the green box
   is only a guide. Keep the camera position the same for all three folders.

## Command (Terminal.app, from the repo root)

```bash
cd ~/projects/IntelRobotChallangeGigglers && source .venv/bin/activate
python vision/capture.py --split training --cls good
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
python vision/train.py --seg scene        # (B) on paper: --seg paper
python vision/predict.py --ckpt results/Patchcore/bricks/v0/weights/lightning/model.ckpt --input dataset/raw/test --seg scene --out results/predict
```

Write down what you see (does the marker light up, do highlights/shadows?) in `docs/experiments.md`.
