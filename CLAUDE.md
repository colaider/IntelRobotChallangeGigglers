# Intel Physical AI Challenge – Team Gigglers (Junction 2026)

Context file for Claude Code. Read this first before working in this repo.

## The challenge (global goal)

Build a closed-loop robotic inspection cell on one station:

> The SO-101 arm picks each Duplo brick, shows it to an inspection camera, an anomaly
> model decides "perfect" vs "defective", and the arm drops it into the matching bowl –
> fully autonomously, with all models optimized with OpenVINO and running on the
> laptop's CPU / GPU / NPU.

Required pipeline (Intel's own wording): **PERCEIVE → DETECT → REASON → ACT → OPTIMIZE**.

- Defects = **marker drawings on the bricks only** (no physical damage).
- A defect can be on **any face** of the brick. Inspecting all faces is our differentiator.
- Exact sorting rules are announced at kickoff / Challenge Day.
- Final: live demo + explanation of architecture and optimization choices.

### Judging (100 pts)

| Criterion | Pts | What it means for us |
|---|---|---|
| End-to-end physical AI solution | 25 | One integrated closed loop, not separate component demos |
| Defect detection with Anomalib | 20 | Detection quality, normal vs defective handling, **how the defect output is consumed by the robot workflow** |
| VLA & Physical AI Studio integration | 20 | Use Intel Physical AI Studio + a VLA policy to turn perception + instruction into actions |
| OpenVINO & Core Ultra optimization | 20 | Latency, throughput, sensible use of NPU / GPU / CPU |
| Robotic execution & reliability | 10 | Accuracy, repeatability, recovery from uncertain detections |
| Innovation & demo | 5 | Originality, clarity of demo and architecture |

Intel's suggested ideas: anomaly score triggers a re-grasp / re-inspection, sharing cameras
between policy and inspection, quantizing models with OpenVINO, few demonstrations,
robustness to new brick positions.

## Hardware & software (per station)

- Laptop: HP OmniBook Ultra or Dell XPS 16, Intel Core Ultra X9 388H (Panther Lake), 64 GB RAM,
  Ubuntu 26.04, NPU + iGPU drivers pre-installed.
- Arm: LeRobot SO-ARM101, 1 leader + 1 follower, wrist camera on the follower.
- Cameras: 1× Intel RealSense D415 (RGB + depth), 1× 2K/60fps webcam.
- Duplo bricks, bowls, markers, desk mounts, USB hub.
- To install ourselves: **Physical AI Studio**, **Anomalib v2.7.0**.
- Cloud GPU for policy training: **Modal** (provided).
- Intel mentors on site; hands-on technical session Saturday 12:00.

## Architecture (planned)

```
 top camera (D415) ──► VLA policy (Physical AI Studio) ──► SO-101 follower
                              ▲           │ pick / regrasp / drop
                              │ instruction ("put it in the defective bowl")
 inspection camera (2K) ──► anomaly service (Anomalib + OpenVINO) ──► web dashboard
                              │ per-face scores, heatmaps, decision
```

- **Physical AI Studio** runs the robot loop (cameras + joint state → policy → joint targets).
  It must stay the system that controls the arm (20 pts depend on it).
- **Anomaly service** (our module): takes inspection frames, returns scores + heatmaps + decision.
  The decision is passed into the Studio workflow (open question: how – see below).
- **Inspection motions**: the brick is picked by the learned policy; the inspection sequence in
  front of the camera may be **fixed joint poses** (repeatable frames = better anomaly detection).
  Must confirm with mentors that this still counts for the VLA criterion.
- **Dashboard** (optional, after the core loop works): live inspection frame, heatmap, per-face
  scores on an unfolded-cube view, decision, cycle time, latency per device.

### Inspection sequence (draft, 7 frames, 2 grasps)

1. Brick on the table → top camera sees the **top** face (studs).
2. Grasp from above on sides C+D, lift → inspection camera sees **side A**.
3. Wrist roll 180° → **side B**.
4. Tilt wrist → **bottom** (cavity).
5. Put down rotated 90°, regrasp on sides A+B.
6. Lift → **side C**.
7. Wrist roll 180° → **side D**.

Brick score = max over face scores. Two thresholds → `good` / `defective` / `uncertain`;
`uncertain` triggers a re-inspection of the worst face. To test: grasping by the corners
(diagonal) might expose all 4 sides with one grasp.

### Anomaly output contract (draft)

```json
{
  "brick": 3,
  "faces": {"top": 0.08, "side_A": 0.11, "bottom": 0.47, "side_B": 0.91, "side_C": 0.09, "side_D": 0.12},
  "max_score": 0.91,
  "worst_face": "side_B",
  "decision": "defective"
}
```

## Current state

- Team formed; Andrii owns **computer vision / Anomalib / OpenVINO / integration / dashboard**.
  Teammates own the arm, teleoperation, demonstrations and VLA training in Physical AI Studio.
- Repo contains `so101_jog.py`: keyboard jogging of the follower via the Feetech STS3215 servo SDK
  (reads and writes raw joint positions). Note: it is **Windows-only** (`msvcrt`); the station
  laptop runs Ubuntu. It is a good base for saving / replaying fixed inspection poses.
- Nothing for Anomalib yet.

## Lessons from first photos of the bricks

- Glossy plastic → specular highlights; hard shadows from a single lamp; inner cavity is dark.
- Wooden table background is noisy (dark knots look like marker dots).
- Fix with: plain matte background (grey/white), diffuse light, light from the camera side for
  the cavity, fixed camera distance, crop the frame to the brick.
- Depth (D415) does not see marker defects → use **RGB (2K webcam)** for inspection. Depth is
  useful only for locating bricks for grasping / background removal.

## Next steps – Anomalib minimal experiment (Andrii, now)

Goal: learn the tool end to end on a tiny dataset before touching the robot.

1. **Environment** (local laptop is fine; PatchCore does not need a cloud GPU):
   ```bash
   python3.12 -m venv .venv && source .venv/bin/activate
   # optional, avoids downloading CUDA builds on a CPU/Intel machine:
   # pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install "anomalib[openvino]==2.7.0"
   ```
   Verify the supported Python version and extras in the Anomalib README if install fails.
2. **Capture script** `vision/capture.py`: OpenCV preview from the webcam, SPACE saves a frame,
   Q quits, saves to `data/bricks/<class>/NNN.png` where class is `good` or `marker`.
   Camera index and resolution as CLI args. Optional fixed center crop.
3. **Collect the minimal dataset** (see "Minimum data" below).
4. **Train** `vision/train.py` with the Folder datamodule:
   ```python
   from anomalib.data import Folder
   from anomalib.models import Patchcore
   from anomalib.engine import Engine

   dm = Folder(name="bricks", root="data/bricks", normal_dir="good", abnormal_dir="marker")
   model = Patchcore()
   engine = Engine()
   engine.train(datamodule=dm, model=model)
   ```
5. **Inspect results**: heatmaps and scores saved by Anomalib (check log for the output folder,
   usually `results/`). Check: marker lights up, highlights/shadows do not.
6. **Break it on purpose**: move the lamp, show an unseen face, tiny dot instead of a line –
   note how scores change. Keep notes in `docs/experiments.md` (useful for the pitch).
7. **Predict script** `vision/predict.py`: single image or folder → score, label, heatmap overlay.
8. **Export to OpenVINO** and benchmark latency per image on CPU / GPU / NPU
   (`vision/benchmark.py`). Final training and benchmarks must run on the **station laptop**.
9. Later: compare PatchCore vs lighter models (e.g. Padim, Stfpm) on accuracy and latency;
   wrap inference in a small service (FastAPI) emitting the JSON contract above.

### Minimum data

- First experiment: **1 colour, 1 face type** (e.g. a side), **20–30 good** images + **5–10 with
  marker**. Small shifts/rotations (±5–10°) like the real pipeline will produce – not random poses.
- Full system: good images are needed **per face type** (top / side / bottom) **per colour**.
  ~20–30 good images per (face type × colour) is a reasonable start. Options for colours:
  one model per colour (pick by a simple HSV colour check) or one model trained on all colours.
- Marker images are only for validation and threshold tuning; training uses good images.
- Use the same camera, light, background and distance as the final inspection booth.

## Open questions for mentors

1. How do we plug an external Anomalib model / decision into the Physical AI Studio workflow
   (runtime config, plugin, separate service)?
2. Do fixed (scripted) inspection poses count, or must inspection be part of the learned policy?
3. Which defects exactly on Challenge Day (marker colours, faces incl. inside the cavity)?
4. How many brick colours/sizes will be used?
5. How to send raw joint targets to the follower inside the Studio setup?

## Conventions

- Python code in `vision/` (anomaly), robot helpers in `robot/`, docs in `docs/`.
- Photos for training/testing live in `dataset/` (committed on purpose, see `dataset/README.md`): `training/good`, `test/good`, `test/marker`. Do not commit `data/`, `results/`, model weights (`*.ckpt`, `*.xml`, `*.bin`).
- Keep scripts runnable with one command and documented in the README (reproducibility matters for the demo).
