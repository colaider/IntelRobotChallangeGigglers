# Inspection plan — 2 cameras (gripper + front), no depth

Handoff document for the next agent. This is the **current team decision** for detection
and dataset collection. Arm kinematics details: [`inspection_sequence.md`](inspection_sequence.md).
Proven vision lessons: [`vision_pipeline.md`](vision_pipeline.md). Experiment numbers:
`results/experiments/grid1/`, `results/experiments/grid2/`.

**Status of data:** the next dataset will be **much cleaner** than `data/` from 2026-11-14
(static backdrop opposite the brick, fewer specular highlights, fixed booth poses). Do **not**
train production models on the noisy hall-background gripper frames; use them only as history.

---

## 1. Goal

Closed loop: pick Duplo brick → inspect all faces for **marker defects** → drop into
good / defective bowl. Early exit on the first **confident** defective face (judge rule).
Uncertain → re-inspect that face before deciding. Brick is good only when every required face
is confidently good.

Hardware used for **detection** (this plan):

| Camera | Role in detection |
|---|---|
| **Gripper (wrist) RGB** | Early reject: if it clearly sees a marker → defective immediately |
| **Front / inspection RGB** (fixed 2K webcam or equivalent) | Main decision: side faces + bottom in fixed booth poses |
| Top (D415 / ground) | **Not used for detection** in this plan (optional later for pick localization only) |
| Depth | **Not used** for detection in this plan |

---

## 2. Runtime flow (per brick)

```
 grasp 1 (jaws on S1+S3)
        │
        ▼
 [G1] gripper RGB  ── score high? ──► DEFECTIVE bowl (skip rest)
        │ no / uncertain
        ▼
 [F1] front: side A (e.g. S2)  ── defective? ──► DEFECTIVE
        │
 [F2] front: side B (roll 180°, e.g. S4)  ── defective? ──► DEFECTIVE
        │
 [F3] front: bottom Bt (gripper horizontal to camera)  ── defective? ──► DEFECTIVE
        │ all good / still uncertain on some
        ▼
 place + regrasp 90° (jaws on S2+S4)
        │
        ▼
 [G2] gripper RGB  ── score high? ──► DEFECTIVE
        │
 [F4] front: side C (e.g. S1)  ── defective? ──► DEFECTIVE
        │
 [F5] front: side D (roll 180°, e.g. S3)  ── defective? ──► DEFECTIVE
        │
        ▼
 all required faces confidently GOOD ──► good bowl
 any remaining UNCERTAIN after re-look ──► policy choice (re-inspect / safer reject)
```

### Rules

1. **First confident defective wins** → stop, defective bowl. Do not wait for other faces.
2. **Gripper is only an accelerator**, not the authority for “good”. A clean gripper view does
   **not** prove the hidden jaw faces are clean → still need regrasp + front.
3. **Front is the authority** for side and bottom verdicts in the booth.
4. **Uncertain** on a face → small re-look (roll ±10–15° or second frame) before treating as good.
   Do **not** mark defective on a soft score (false reject cannot be undone).
5. Faces under the jaws are invisible; never ask a model to score a gripped face.

### What each camera sees in typical poses

| Step | Gripper cam typically sees | Front cam sees |
|---|---|---|
| After lift, jaws down | Top (T) between jaws / palm gap (partial, useful for early reject) | Side face square-on |
| Horizontal to front | Still roughly top / clutter | Bottom (Bt) |
| After regrasp | Again top-ish early reject | The two sides that were hidden before |

Do **not** assume the top (overhead) camera sees the brick top while the front is inspecting a
side: the gripper body sits above the brick and blocks it.

---

## 3. Models — how to organize

### Principle

Retrain whenever **camera + background + lighting + framing** change. A table-trained model
flags every gripper frame as defective (proven). Finder/crop choice is **preprocessing**, not
a separate model family.

### Recommended model set (OpenVINO PatchCore)

| ID | Camera | View | Trained on | Used at |
|---|---|---|---|---|
| `gripper_early` | gripper | whatever is visible in grasp pose (mostly top) | good crops from gripper in **booth**, same window | G1, G2 |
| `front_side` | front | side face square-on | good side crops from front booth pose | F1, F2, F4, F5 |
| `front_bottom` | front | cavity / bottom | good bottom crops from horizontal pose | F3 |

Optional later:

- `front_top` — only if you deliberately show top to the front camera (not in default sequence).
- Per-colour banks (`…_yellow`, `…_red`, …) or HSV router → colour-specific weights if one bank
  dilutes (grid2: `per_face_colour` helped on ground).

### Do **not**

- One shared model for gripper + front.
- One `all` model mixing top / side / bottom for the front booth.
- A separate model per finder name (`window` vs `bg`) — pick **one** crop pipeline per camera.
- Train on hall / stage lighting frames for the production booth models.

### Defaults from experiments (noisy old data — re-validate on clean set)

- Model: **PatchCore** (beats PaDiM here).
- CLAHE: prefer **off** unless clean-data ablations say otherwise.
- Gripper crop: **`window`** (fixed ROI between jaws + segment inside). `window_raw` / `full` failed.
- Ground/front fixed camera crop (old data): `bg` or `bg_scene`; on the **new static backdrop**,
  prefer fixed **`window`** at the booth pose (faster, more repeatable). Keep `bg` as fallback
  if the brick drifts in frame.
- Score for decision: prefer **max anomaly on eroded brick mask** (“brick score”), not raw
  image score, so gripper/background rim does not dominate.
- Metrics that matter: AUROC + **caught@0FA** (defects above every good test crop) + dual
  thresholds (good / defective / uncertain).

### Inference service contract (target)

One service; request selects weights:

```json
{
  "camera": "gripper" | "front",
  "view": "early" | "side" | "bottom",
  "image": "<bgr frame or crop>",
  "brick_id": 3
}
```

Response (align with team JSON draft):

```json
{
  "score": 0.81,
  "decision": "defective" | "good" | "uncertain",
  "camera": "front",
  "view": "side",
  "heatmap_path": "optional"
}
```

Brick-level aggregator: keeps per-face scores; on first `defective` emits final brick decision;
else after all faces emits `good` or `uncertain`.

---

## 4. Crop / preprocess (no depth)

Per camera, fixed pipeline at train **and** runtime:

### Gripper

1. Fixed fractional window around jaws (see `vision/finders.py` `WINDOWS["gripper"]`;
   **re-measure** on the clean booth contact sheet and update constants).
2. Segment brick inside the window (`scene` / GrabCut path in `segment.py`).
3. Drop obvious jaw leftovers; `mask_quality` fail → `uncertain`, do not score as good.
4. Canonical square crop, grey fill, resize 256×256 → model.

### Front (booth)

1. Fixed window for the inspection TCP pose (brick always in the same place if poses are fixed).
2. Segment inside window (or tight window_raw if backdrop is truly uniform and QA is clean).
3. Same canonical crop → 256×256.

Latency: avoid full-frame `scene` (~1 s). Windowed search is mandatory for cycle time.

---

## 5. New dataset — what to collect

### Setup (lock before first photo)

- Static **matte** backdrop opposite the inspection spot (grey/white board or light tent).
- Diffuse light, camera-side fill for the cavity; minimise specular blobs on studs.
- Front camera fixed; booth TCP poses fixed (same as runtime).
- Gripper photos also in that booth (same light/background), not mid-air over the hall.
- Do not change lamp/camera between train and test.

### Splits

| Split | Content |
|---|---|
| Train | **Only good** bricks |
| Test good | Good, different pose groups / instances (not near-duplicates of train) |
| Test defective | Real marker defects for thresholding (line, stroke, dot, tiny 1–2 mm, near edge) |

Synthetic defects in `vision/experiment.py` are for offline comparison only; final thresholds
need real markers.

### Volumes (minimum useful)

Per **colour** × **view** (`gripper_early`, `front_side`, `front_bottom`):

- Train good: **20–30** frames  
- Test good: **≥10**  
- Test defective: **≥10** (mix of defect kinds)

Between good frames: shift ±5–10°, small translations; keep face type and distance stable.

### Poses that matter (shoot these)

| Label | How | Cameras to save |
|---|---|---|
| `gripper_early` | After grasp, at booth height, jaws down | gripper RGB |
| `front_side` | Booth pose, face square to front cam | front RGB |
| `front_bottom` | Gripper horizontal toward front cam | front RGB |
| (optional) second grasp repeats for the other two sides | same as `front_side` | front RGB |

If sessions are recorded as multi-camera dumps (like current `data/`), keep naming consistent
and label **face / view** in `dataset/faces.csv` (or successor). Gripper frames used for
`gripper_early` must be tagged so they are not mixed into `front_side` training.

### Poses / frames to discard (do not train on)

- Hall / people / stage purple light in background  
- Motion blur, out of focus  
- Strongly skewed “almost corner” views  
- Mid-trajectory frames between poses  
- Faces currently under the jaws  
- Mixed setups in one model folder (old table + new booth)

### QA before training

For each camera × finder, generate a contact sheet (`experiment.py --qa` or equivalent).

Accept only if:

- Brick fills most of the crop, roughly centered  
- Mask edges are stable across the sheet  
- Backdrop is uniform; jaws only at borders  
- Failures are flagged `ok=false` and excluded from train (or fixed)

Update `WINDOWS` / booth window from these sheets.

---

## 6. Decision thresholds

Per model (`gripper_early`, `front_side`, `front_bottom`):

1. Score all test-good and test-defective crops.  
2. Set **high** (defective) so false alarms on good ≈ 0 (caught@0FA style).  
3. Set **low** (good) below the bulk of good scores.  
4. Band between = **uncertain** → re-look.

Gripper high threshold should be **stricter** than front if needed: gripper is early-reject only;
a false defective from glare is costly.

Brick score = max over face decisions (with early exit).

---

## 7. Software map (repo)

| Piece | Path / note |
|---|---|
| Finders / windows | `vision/finders.py` |
| Segment / canonical crop | `vision/segment.py` |
| Session loading | `vision/sessions.py` |
| Compare grid / QA | `vision/experiment.py` → `results/experiments/` |
| Train / predict / export | `vision/train.py`, `predict.py`, `export.py`, `benchmark.py` |
| Arm pose narrative | `docs/inspection_sequence.md` |
| Older photo guide | `docs/photo-guide.md` (partially outdated; prefer this doc) |

Suggested next implementation steps for the next agent:

1. Freeze booth windows from the **new** clean contact sheets; commit updated `WINDOWS`.  
2. Collection script or session convention: folders or tags per `camera` × `view` × `colour` ×
   `split` (`good` / `defective`).  
3. Train three PatchCore models; export OpenVINO; benchmark CPU/GPU/NPU on station laptop.  
4. Small FastAPI (or similar) anomaly service with the JSON contract above.  
5. Wire robot loop: after each fixed pose, call service with `(camera, view)`; honor early exit.  
6. Calibrate dual thresholds on real markers; log scores for the demo dashboard.

---

## 8. Evidence snapshot (why this plan)

From `grid1` / `grid2` on the **old noisy** set (expect better numbers on clean data):

- Gripper **`window` + PatchCore**: best early signal (~AUROC 0.85, caught@0FA ~30/72) vs
  `full` / `window_raw` (~0.5).  
- Crop QA for gripper `window` still weak (`ok` ~33/92) → clean backdrop + remeasured window
  is the main lever, not more model shopping.  
- PatchCore ≫ PaDiM; CLAHE not clearly helpful.  
- `per_face` / `per_face_colour` beats dumping all views into one bank.  
- Segmentation / ROI is the biggest historical win (`vision_pipeline.md`).

---

## 9. One-paragraph summary for humans

We inspect with **two RGB cameras**: the **gripper** only for **confident early reject**, the
**fixed front camera** for **side and bottom** in a short booth sequence with one **regrasp** so
all four sides are seen. **No depth, no top-cam detection.** Separate PatchCore OpenVINO models
per camera/view (`gripper_early`, `front_side`, `front_bottom`), same crop pipeline at train and
runtime, dual thresholds with uncertain re-look. Next dataset is shot on a **static backdrop**
with **fewer highlights** in the **exact runtime poses**; old hall-background data is obsolete
for production weights.
