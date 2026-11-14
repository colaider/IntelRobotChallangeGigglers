# Dataset workflow — collect → label → build views → train → calibrate → inspect

For any developer (or their coding agent) who collects a new dataset. Follow the steps in order;
every command runs from the repo root. Why the steps look like this:
[`inspection_plan_2cam.md`](inspection_plan_2cam.md) (team plan) and
[`vision_pipeline.md`](vision_pipeline.md) (what is proven).

```
 collection/run.py  ──►  data/<session>/        raw frames of every camera + depth + arm pose
        │
 vision/sessions.py --sheet  ──►  dataset/faces.csv     which face the front camera sees (label by hand)
        │
 vision/build_views.py  ──►  dataset/views/<view>/{training/good, test/good, test/defective}
        │                    brick already cut out; look at qa.jpg
 vision/train.py --root dataset/views/<view>        one PatchCore model per view
 vision/export.py   ──►  OpenVINO IR (FP16/INT8)
 vision/calibrate.py ──► thresholds low/high (JSON)
 vision/inspect_brick.py ──► good / defective / uncertain per face, early exit per brick
```

Views (one model each): `gripper_top` (gripper camera, top face after the grasp), `front_side`,
`front_bottom` (front camera = RealSense `ground`), optional `front_top`.

---

## 1. Collect (`collection/run.py`)

Full instructions: [`collection/README.md`](../collection/README.md). Must-dos:

- **Two Python environments:** the collector runs with the *system* Python
  (`py -m pip install -r collection/requirements.txt`), training with the repo `.venv`
  (`vision/README.md` → Setup). Close Physical AI Studio / `so101_jog.py` first (COM port, cameras).
- **Lock the setup before the first frame:** static matte backdrop behind the inspection spot,
  diffuse light (shield room LEDs), press `a` once so exposure + white balance are fixed for the
  whole session, camera never moves. One session = one sitting; new setup → new session.
- **Shoot in the exact runtime poses** (booth poses of the arm): after grasp from above
  (gripper sees the top), side face square to the front camera, roll 180° for the other side,
  gripper horizontal for the bottom; then regrasp 90° and repeat (`inspection_sequence.md`).
  Between good frames: small rotations ±5–10° / shifts, same distance.
- **Session names** `<colour>_<label>_<letter>`, e.g. `red_good_a`, `red_good_b` (test-only),
  `red_defective_a`. Keep **at least one good session per colour only for testing**.
- **Defective sessions:** draw real marker defects (line, stroke, dot, tiny 1–2 mm dot, near the
  edge); label with `d` + defect number in the collector.
- **Discard** (BACKSPACE): motion blur, fingers on the brick, brick cut by the frame, mid-move frames.
- **Resolution:** collect at the resolution you will run at (default 640×480 everywhere). At
  640×480 a brick is ~150–250 px wide → fine for ≥2 mm marks, borderline for 1 mm dots. If tiny
  dots matter, try one session at 1280×720 for the front camera and compare before switching
  (never mix resolutions in one model).
- **Depth:** saved correctly (16-bit, millimetres; it only *looks* black in an image viewer). The
  front camera is closer to the brick than the D415 minimum range (~30 cm), so there is **no depth
  on the brick** → not usable for the inspection mask there. Top camera depth is fine (pick).

Volumes per colour × view: **train good 20–30, test good ≥10, test defective ≥10.**
Colour-specific models are postponed; start with one model per view, all colours.

## 2. Label faces (`dataset/faces.csv`)

The front camera can see top / bottom / side, so every capture gets a face label:

```bash
.venv/bin/python vision/sessions.py --sheet ground gripper     # numbered contact sheets per session
```

Add one row per capture to `dataset/faces.csv` (`session,id,face`, face ∈ `top bottom side none
occluded`). `none` = no brick in the front view, `occluded` = fingers/cut off (both skipped).
The gripper view needs no label (it always sees the top after the grasp).

## 3. Share the data

`data/` is gitignored (hundreds of MB of PNG + depth). Share sessions as a zip / shared drive and
unpack into `data/`; commit only `dataset/faces.csv` and code. (Git LFS is an option if the team
wants the raw sessions in the repo.)

## 4. Build per-view datasets

```bash
.venv/bin/python vision/build_views.py                  # all views -> dataset/views/<view>/
```

- Cuts the brick out per camera (`gripper`: fixed window between the jaws + segmentation;
  front: background difference against the median empty scene) — same finders as at runtime.
- Splits good frames into train / test **by arm-pose group and colour** (near-duplicates never
  leak into test); frames of `defective` sessions go to `test/defective`.
- **Look at `dataset/views/<view>/qa.jpg`** (every crop, labelled train/test) and `skipped.txt`
  (crops the quality check rejected — they are not used). Bad crops → fix the setup or
  `vision/finders.py` (`WINDOWS["gripper"]` window) before training.

## 5. Train, export, calibrate (per view)

```bash
V=front_side
.venv/bin/python vision/train.py  --root dataset/views/$V --results results/$V
.venv/bin/python vision/export.py --root dataset/views/$V --out results/$V/export \
    --ckpt results/$V/Patchcore/bricks/v0/weights/lightning/model.ckpt
.venv/bin/python vision/calibrate.py --seg none --xml results/$V/export/patchcore/fp16/weights/openvino/model.xml \
    --good dataset/views/$V/test/good --defective dataset/views/$V/test/defective \
    --out results/$V/calibration.json
```

- No `--seg` here: the crops are already brick-only.
- Without real `test/defective` frames, `train.py` falls back to Anomalib's synthetic anomalies and
  says so — metrics and thresholds are then indicative only.
- `calibrate.py` warns when defective frames score below `low` (not separable → more data / light).

## 6. Check

```bash
.venv/bin/python vision/inspect_brick.py --seg none --xml results/$V/export/patchcore/fp16/weights/openvino/model.xml \
    --calib results/$V/calibration.json --folder dataset/views/$V/test
.venv/bin/python vision/benchmark.py --root dataset/views/$V --models results/$V/export --devices CPU GPU NPU
```

Write what you see (scores, misses, false alarms, latency per device) into `docs/experiments.md`.

## Checklist for a new dataset

- [ ] backdrop + light fixed, exposure/WB locked (`a`), camera not moved
- [ ] frames in the runtime booth poses only; blurred / occluded ones deleted
- [ ] per colour: ≥2 good sessions (one test-only) + 1 defective session with real marks
- [ ] `dataset/faces.csv` filled for every new capture
- [ ] `build_views.py` run, `qa.jpg` looked at, `skipped.txt` reasonable
- [ ] per view: train → export → calibrate → inspect → numbers in `docs/experiments.md`

Known gaps (to adapt when the clean booth data arrives): the gripper window in `finders.py` was
measured on the old hall frames (many crops rejected there), front-camera booth window not set
yet (background difference is used), and `inspect_brick.py` runs one model at a time — a
per-(camera, view) model bank comes next.
