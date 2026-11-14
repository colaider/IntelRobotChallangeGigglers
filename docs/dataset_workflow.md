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
 vision/build_views.py  ──►  dataset/views/<camera>_<face>, <camera>_all     brick cut out; look at qa.jpg
        │
 vision/train_views.py  ──►  every variant: train → OpenVINO export → thresholds
        │                    results/models/models.json (registry) + compare.md (single vs per_face)
 vision/train_face_classifier.py (optional) ──► face classifier per camera, added to the registry
        │
 vision/inspector_bank.py  ──►  per face: cut out → face → score → good / defective / uncertain;
                               per brick: early exit, faces seen / missing; mode single | per_face | classifier
```

Cameras are never mixed. Per camera: `<camera>_top`, `<camera>_bottom`, `<camera>_side` (one model per
face) and `<camera>_all` (one model for all faces) — cameras `gripper` and `front` (= RealSense `ground`).
The brick's starting orientation is unknown, so any face can reach either camera: shoot and label
every face for both cameras.

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

Both cameras can see top / bottom / side, so every capture gets two labels:

```bash
.venv/bin/python vision/sessions.py --add-missing             # adds empty rows for new captures
.venv/bin/python vision/sessions.py --sheet ground gripper    # numbered contact sheets per session
```

Fill `dataset/faces.csv` (`session,id,face,gripper_face`): `face` = what the front camera sees,
`gripper_face` = what the gripper camera sees; values `top bottom side none occluded` (`none` = no
brick in that view, `occluded` = fingers / cut off — skipped). Empty `gripper_face` counts as `top`.

## 3. Share the data

`data/` is gitignored (hundreds of MB of PNG + depth). Share sessions as a zip / shared drive and
unpack into `data/`; commit only `dataset/faces.csv` and code. (Git LFS is an option if the team
wants the raw sessions in the repo.)

## 4. Build the datasets

```bash
.venv/bin/python vision/build_views.py          # -> dataset/views/<camera>_<face> and <camera>_all
```

- Cuts the brick out per camera (`gripper`: fixed window between the jaws + segmentation; `front`:
  difference to the median empty scene, saved as `dataset/views/front_background.png` because the
  runtime needs the same reference) — the same finders as at runtime.
- Splits good frames into train / test **by arm-pose group and colour**, the same split for every view
  of a camera; frames of `defective` sessions go to `test/defective`.
- **Look at `dataset/views/<view>/qa.jpg`** and `skipped.txt` (crops rejected by the quality check).

## 5. Train every variant (one command)

```bash
.venv/bin/python vision/train_views.py                    # single + per_face, both cameras
.venv/bin/python vision/train_face_classifier.py          # optional: face classifier per camera
```

- Per view: PatchCore training (synthetic anomalies if no real `test/defective`), OpenVINO FP16 export
  (`--int8` for INT8 too), thresholds `low` / `high` from the clean test crops and the defective ones
  (real if present, else synthetic marker defects — indicative).
- Writes `results/models/models.json` (registry: camera → finder, background, models, thresholds,
  classifier) and `results/models/compare.md`: **single vs per_face** per camera (AUROC, caught@0FA,
  how often the competition picked the right face). Decide the default mode from this table.

## 6. Check / use

```bash
.venv/bin/python vision/inspector_bank.py --camera front --mode per_face \
    --folder data/<session> --pattern "*_ground_color.png"
.venv/bin/python vision/benchmark.py --root dataset/views/front_side --models results/models/front_side/export \
    --devices CPU GPU NPU
```

Robot code:

```python
from inspector_bank import InspectorBank
bank = InspectorBank("results/models/models.json", mode="per_face", device="NPU")   # or single / classifier
res = bank.inspect_brick(lambda step, attempt: grab(step), steps=[("G1", "gripper"), ("F1", "front"), ...])
res["decision"]    # good / defective / uncertain / incomplete (+ res["missing"] faces)
```

Write what you see into `docs/experiments.md`.

## Checklist for a new dataset

- [ ] backdrop + light fixed, exposure/WB locked (`a`), camera not moved
- [ ] frames in the runtime booth poses only; blurred / occluded ones deleted
- [ ] per colour: ≥2 good sessions (one test-only) + 1 defective session with real marks
- [ ] `dataset/faces.csv` filled for every new capture
- [ ] `build_views.py` run, `qa.jpg` looked at, `skipped.txt` reasonable
- [ ] `train_views.py` run, `compare.md` read (single vs per_face), default mode chosen
- [ ] numbers in `docs/experiments.md`

Known gaps (to adapt when the clean booth data arrives): the gripper window in `finders.py` was
measured on the old hall frames (many crops rejected there) and the front camera uses the
background difference (a fixed booth window would be faster). On the old data the face competition
picked the right face only ~50 % of the time (all models score high on noisy frames) while the
classifier got 9/9 — re-check both on the clean data before choosing the default mode.
