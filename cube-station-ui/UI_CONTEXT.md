# UI context: cube pick / inspect / sort station

A brief for whoever builds the operator UI (a person or an AI agent). It explains what the system does, what exists today, where a UI can connect, and the safety rules the UI must keep.
State as of 2026-11-14. All code is in `robot_sequence/`, and `README.md` there has the details.

---

## 1. What the station does

An SO-101 robot arm (follower) runs a vision-language-action policy (SmolVLA, maybe pi0.5 later) and a defect-inspection model (PatchCore). It handles one cube at a time:

```
STANDBY --g--> PICK -> SNAP 1 -> SIDE2 -> SNAP 2 -> ROLL +180 -> SNAP 3 -> DECIDE
           -> SORT (right bowl = good | left bowl = defective) -> RELEASE -> HOME -> STANDBY
```

| step type | meaning | how it ends |
|---|---|---|
| **VLA** | the model drives the arm from a text prompt plus 3 camera images plus joint state | the "done" rules (cube still, at the end pose, model wants to stay) or the operator (manual mode) |
| **SCRIPT** | scripted joint moves: roll the wrist 180°, open/close the gripper, go home | target reached |
| **SNAP** | arm holds still; one frame per camera; the cube is found, cropped and scored | verdict good / defective / uncertain |
| **DECIDE** | the cube is good only if all 3 faces were seen and good | immediate |

Any failure (timeout, stuck, tracking error, operator pressed **q**, exception) leads to **STANDBY**. The arm holds its pose with torque on; nothing moves until the operator chooses.

**Who uses the UI:**
- the operator at the station during the hackathon demo (non-expert; must stay safe);
- the team while debugging and teaching the done rules.

---

## 2. What exists today (no UI, terminal only)

| program | purpose |
|---|---|
| `robot_loop.py` | full state machine, loops over cubes |
| `debug_sequence.py` | fixed step list run one step at a time (`--snap-test` = PICK → SNAP → HOME along the recorded path) |
| `derive_done_rules.py` | learns the done-rule thresholds from operator-ended steps and writes `done_rules.json` |
| `verify_inspection.py` | offline check of the inspection crop and scoring on recorded data |
| `--sim` flag (both runners) | full offline simulator: fake arm, cameras and models. **Use this to develop the UI without hardware.** |

**Keyboard controls** (single keys in the terminal) are the whole control surface the UI must offer:

| context | keys |
|---|---|
| any moving step | **q** hold now (step fails → STANDBY), **s** stop after this cube |
| manual VLA step | **n** / **space** = "this step is finished" (then 1.5 s more recording) |
| STANDBY (`robot_loop`) | **g** next cube (only near the START POSE), **r** slow move to the START POSE, **b** back along the failed step's path, **t** retry the failed step, **x** exit |
| before a step (`debug_sequence`) | **g** run, **k** skip, **x** stop |
| after a failed step (`debug_sequence`) | **b** back along the path, **r** retry, **k** skip, **x** stop |

---

## 3. Places a UI can connect (no rewrite needed)

| hook | where | what it gives |
|---|---|---|
| **Commands** | `Keys.feed(key: str)` in `robot_loop.py` | the UI sends the same keys as the keyboard. It's thread-safe, and the keyboard keeps working at the same time. |
| **Event lines** | `Sequence.say(msg)`, appended to `Sequence.events` | human-readable progress lines (step start/end, snaps, decisions, failures, STANDBY with its options) |
| **Live step signals** | `run_prompt(..., status=callback)` in `policy.py`, called about every 0.5 s in manual mode | t, pose_err, cube trend (mm/s), wobble (mm), flat chunks, still time, "rules say: moving / DONE / STUCK" |
| **Per-tick step log** | `StepResult.log` (30 Hz rows) | `t, q[6], cmd[6], chunk_motion, pose, plan, still_s, range, trend, wobble, tip[xyz mm], pose_err, pose_err_j[5], done, stuck` |
| **Step result** | `StepResult` / `.summary` | reason (DONE / OPERATOR), duration, chunks, rejected chunks, conditioner correction, model latency, when the rules would have fired (`auto_done_t`, `auto_stuck_t`, `timeout_t`), `t_operator` |
| **Snap result** | `inspection.Snap` | face, verdict, score, all camera frames (RGB), crop the model saw (RGB), anomaly heatmap (0..1), `find` = {ok, reasons, ms} |
| **State** | `Sequence` fields | `cube_n`, `snaps`, `verdict`, `standby_reason`, `failed_state`, `start_pose`, `run_dir`, `report.data` |
| **Arm** | `hw.Robot` | `read()` joints (normalized), `last_sent`, `temperatures()` (°C per servo) |
| **Cameras** | `hw.CameraHub.cameras[name].read_latest()` | newest frame, non-destructive. `CameraHub.latest()` / `fresh()` are reserved for the current owner (VLA / INSPECT); a preview must use `read_latest()` and must not change `owner`. |
| **Kinematics** | `kinematics.kinematics().tool(q)` / `.cube_points(q)` | tool pose in mm, to draw the arm or the cube trajectory |

The events are plain strings today. A UI should add a small structured event emitter next to `say()` (see section 5) instead of parsing text.

---

## 4. Suggested architecture

```
 browser (localhost)  <--WebSocket events / REST commands / MJPEG-->  ui_bridge (FastAPI + uvicorn thread)
                                                                          │ same process
                                                                          ▼
                                                     Sequence / run_prompt / InspectorBank / CameraHub / Robot
```

- **Same Python process** as the runner. The robot loop stays on the main thread, and FastAPI/uvicorn runs in a daemon thread. Both are installed in the backend venv (fastapi 0.143, uvicorn 0.54): `~/intel-challenge/physical-ai-studio/application/backend/.venv/bin/python`.
- **Port `8765`, bound to `127.0.0.1` only.** Physical AI Studio already uses 7860.
- **The UI is optional.** If the browser closes or the bridge crashes, the robot loop and the keyboard keep working. A bridge exception must never stop or move the arm.
- **Frontend:** one static page served by the bridge (no build step), or a small React/Vite app. Must work with `--sim`.
- **Camera previews:** MJPEG endpoints `/cam/{name}` at a low rate (5–10 fps) via `read_latest()`. Don't open the cameras again; they're opened once by `CameraHub`.

---

## 5. Proposed event and command protocol

**Events (server → UI, JSON over WebSocket):**

| type | fields |
|---|---|
| `hello` | mode (`loop` / `debug` / `sim`), model path, steps list, config summary |
| `state` | state (enum `S` value), cube_n, options (keys allowed now, with labels) |
| `step_started` | step, kind (VLA / SCRIPT / SNAP), prompt, attempt, timeout_s, rules_source |
| `step_status` | t, pose_err, trend_mm_s, wobble_mm, plan_flat, still_s, rules_say, manual (about 2–5 Hz) |
| `step_finished` | step, ok, reason, duration_s, chunks, rejected, max_correction, latency_ms, auto_done_t, t_operator |
| `snap` | face, verdict, score, low, high, find {ok, reasons, ms}, image URLs (crop, heatmap, frames) |
| `decision` | cube_n, verdict, reason, faces[] |
| `failure` | step, code (F1 / STUCK / S4 / R1 / R2 / C2 / MODEL / …), message, options |
| `temps` | per-servo °C (once per cube) |
| `log` | the raw `say()` line (keep it: the terminal text is the ground truth) |
| `run_done` | run_dir, cubes, good / defective counts |

**Commands (UI → server):** `POST /cmd {"key": "g"}`. The server checks the key against the current `options` and forwards it to `Keys.feed`. The **Hold** button sends `q`.

**Read-only REST:**
- `/api/runs`, `/api/runs/{run}/cubes/{n}` (`cube.json` plus images)
- `/api/done_rules`, `/api/prompt_stats`, `/api/config`

---

## 6. Screens

1. **Operator dashboard (main).**
   - State-machine strip with the current step highlighted.
   - **Big red HOLD button,** always visible, with keyboard **q** too.
   - Context buttons from `options`: Start next cube / Back / Retry / Home / Stop after this cube / Exit.
   - Three live camera tiles.
   - Current-step panel: prompt, time against timeout, live charts of cube trend (mm/s) and wobble (mm) with their thresholds, pose error, "rules say".
   - Inspection panel: latest crop, heatmap, and a score bar with the good (<low) / uncertain / defective (≥high) bands.
   - Cube counter (good / defective), servo temperatures (warn ≥ 60 °C), event log.
2. **Teach mode** (`debug_sequence` manual).
   - Big **"Step finished"** button, which sends `n`.
   - After the press: a 1.5 s "recording" indicator.
   - Count of recorded decisions per prompt (`done_logs/steps.jsonl`).
   - Button to run `derive_done_rules.py` and show its report: agree / early / late / never / stuck, the chosen thresholds, and "Write rules" (`--write`, with confirmation).
3. **Snap test** (`--snap-test`): PICK → SNAP → HOME. Shows the detection result large: verdict, score, crop, heatmap, how long the crop took.
4. **Run review:** browse `runs/<time>/cube_NNN/`: images, scores, step times and errors per cube. Plot a step's per-tick log (trend, wobble, pose_err over time, with the operator press and the rule firing marked).
5. **Settings (read-only first):** the key values from `config.py` and `done_rules.json`. Editing config can come later, and must never touch servo or calibration values.

---

## 7. Data on disk

| path | content |
|---|---|
| `runs/<YYYYmmddTHHMMSS>/events.log` | every `say()` line of the run |
| `runs/<…>/front_background_live.png` | background captured at startup (inspection) |
| `runs/<…>/cube_NNN/cube.json` | `{cube, started, finished, verdict, reason, steps:[{name, kind, duration_s, ok, reason, chunks, rejected, max_correction, attempt, auto_done_t, …, error}], snaps:[{face, attempt, score, verdict, find}]}` |
| `runs/<…>/cube_NNN/face{F}_try{A}_{camera}.jpg`, `_crop.png`, `_heatmap.png` | snap images |
| `runs/<…>/cube_NNN/log_{NN}_{prompt}.json` | per-tick log of a VLA step (fields in section 3) |
| `done_logs/steps.jsonl` | one line per operator-ended step: time, model, prompt, t_operator, t_end, auto_done_t, auto_stuck_t, start_q, end_q, latency_ms, log file name |
| `done_rules.json` | derived per-prompt rules: end_mean, tol, timeout_s, still_method, still_trend, still_wobble, trend_window_s, plan_flat, done_still_s, runs, evidence, replay |
| `prompt_stats.json` | per prompt from the recordings: start/end pose mean and std, action range, length, speed limits |

---

## 8. Domain facts the UI must get right

- **Joint units:** body joints −100…+100, gripper 0…100 (Studio's normalized units, not degrees). Joint order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
- **Prompts are exact strings:**
  - "Pick up the cube and show front side to the Side Camera"
  - "Show side 2 to the Side Camera"
  - "move cube to the right bowl" (good)
  - "move cube to the left bowl" (defective)
- **Inspection thresholds** (`front_all` model): good < 0.825 ≤ uncertain < 0.975 ≤ defective. Uncertain is re-snapped once; still uncertain → defective. "No cube found" counts as uncertain.
- **Stillness:** the cube counts as "still" by its motion in mm (forward kinematics), not by joint units: trend below X mm/s **and** wobble below Y mm.
- **Model speed:** model chunk = 50 steps at 30 Hz; inference about 230 ms on the Arc GPU; model load about 20 s at startup.
- **Cameras:** `camera top` (RealSense), `camera gripper`, `camera side` (inspection camera), all 640×480 at 30 fps.

---

## 9. Safety rules for the UI (non-negotiable)

1. **HOLD (q) must work at any time,** from the UI and from the keyboard, and needs no confirmation. The physical power switch stays the real emergency stop; show a reminder.
2. **No motion on its own.** Opening, reloading or reconnecting the UI must never send a key. Only explicit button presses send commands.
3. **Show only the commands allowed in the current state.** The server validates again anyway.
4. **Ask for confirmation before motion from STANDBY:** Start, Home (r), Back (b), Retry (t). The confirmation says what will move.
5. **Bind to localhost only,** with no remote control.
6. **The UI never writes servo registers, EEPROM, calibration or Studio data,** and never starts training.
7. **If the bridge fails, the loop keeps running** with keyboard control. A UI error must not raise into the robot loop.
8. **Camera previews never take camera ownership,** and never run while Physical AI Studio holds the cameras (the runner already refuses to start then).

---

## 10. Open questions for the UI

- Start the runner from the UI (choose model, mode, sim), or only attach to a runner started in the terminal? Attaching is simpler and safer for the first version.
- Show a 3D arm view? Studio already has a URDF viewer (three.js / urdf-loader in `application/ui`); `kinematics.py` gives the same angles.
- Multiple viewers (judges watching on another laptop)? That would need read-only remote access later. Keep localhost-only for the demo.
