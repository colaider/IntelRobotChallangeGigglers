# Experiments

## 2026-11-14 — first real dataset (MacBook, CPU only)

Data (commit e069fcc): 41 `training/good`, 12 `test/good`, 8 `test/defective`, 700×700 crops,
3 colours mixed (lilac, yellow, red), one side face, brick fills/overflows the frame.

### PatchCore (default, 256×256), OpenVINO on Apple M-series CPU

| precision | p50 ms | p95 ms | fps | AUROC | F1 | size MB |
|---|---|---|---|---|---|---|
| FP32 | 54.6 | 65.1 | 18.6 | 0.875 | 0.737 | 125 |
| FP16 | 54.5 | 65.8 | 18.2 | 0.875 | 0.737 | 63 |
| INT8 (NNCF PTQ) | 48.6 | 60.3 | 20.8 | 0.865 | 0.750 | 58 |

Latency = resize + pre/post-processing + inference (what the robot loop sees). AUROC/F1 on all
20 test images. INT8 is ~11% faster with no real accuracy loss; size barely drops because the
PatchCore memory bank dominates. Station numbers (CPU/GPU/NPU) still to do.

### Model comparison (FP16, CPU)

| model | AUROC | F1 | p50 ms | MB | train s |
|---|---|---|---|---|---|
| PatchCore | 0.833 | 0.700 | 57.1 | 63 | 18 |
| PaDiM | 0.792 | 0.545 | 25.0 | 88 | 3 |
| STFPM (20 ep) | 0.615 | 0.600 | 39.3 | 11 | 58 |

(Run-to-run AUROC differs slightly from the table above: separate training runs.)

### Synthetic defects (72 on test/good, scored by the PatchCore above)

| kind | detected | mean score |
|---|---|---|
| curve | 11/11 | 0.97 |
| scribble | 10/10 | 0.96 |
| line | 14/14 | 0.89 |
| dot | 16/16 | 0.81 |
| edge | 10/11 | 0.73 |
| tiny (1–2 mm) | 7/10 | 0.52 |

### Observations

- Big marker strokes (lines, X, letters) → score ~0.9–1.0, heatmap exactly on the ink.
- Misses: tiny dots at the edge of the lilac brick (score ≤ 0.50).
- False positives (~0.55): **specular highlights** from the lamp (white streaks, left side) and
  unusual brick positions (brick half out of frame).
- Next: diffuse light / polariser, brick fully inside the crop with margin, one colour per model
  or more good images per colour, more `test/defective` with small dots; tune threshold
  (currently 0.5 from Anomalib's F1-adaptive threshold on half of `test/`).

## 2026-11-14 — second dataset (Ivan, 01fd167), augmentation test (MacBook, CPU)

Data: 40 `training/good` (yellow 22, red 9, lilac 9), 21 `test/good`, 20 `test/defective`
(mostly marker dots). Closer framing, varied angles; many frames are **out of focus**.

### PatchCore: base vs `--augment 3` vs `--augment 3 --aug-flip` (3 seeds each)

| config | AUROC (Anomalib test half, 3 seeds) | F1 (3 seeds) | AUROC / F1 all 41 imgs (OpenVINO FP16) | real FP / FN (all 41) | size MB |
|---|---|---|---|---|---|
| base | 1.00 / 1.00 / 1.00 | 0.82 / 0.82 / 0.89 | 0.943 / 0.865 | 0/21 · 5/20 | 63 |
| aug3 | 1.00 / 1.00 / 1.00 | 0.89 / 0.82 / 0.82 | 0.952 / 0.889 | 0/21 · 4/20 | 100 |
| aug3 + flip | 1.00 / 0.99 / 1.00 | 0.89 / 0.82 / 0.89 | 0.957 / 0.872 | 1/21 · 4/20 | 100 |

- The 4 defects every model misses (`021352_c700_000/001/005/019`) show **no visible marker** even at
  full resolution (blurred frames or the mark is on a hidden face); `_010` is a cavity view (0.495 base).
  On defects that are actually visible, the base model has no misses and no false alarms.
  → relabel / reshoot those frames and fix focus before trusting F1.
- Augmentation: small gain at best (AUROC +0.01, one fewer borderline miss), within run-to-run noise.
  Costs +60% memory bank / model size unless `--coreset-ratio 0.1/(N+1)` is used. Flip adds a false alarm.
  → keep augmentation off by default; re-test on a cleaner dataset.
- Latency here ~90–105 ms p50 (vs ~45 ms in the first run) – the Mac was busy with other jobs; not comparable.

### Sensitivity (synthetic round dots on the 21 clean test bricks, base model)

Detected share; bricks whose clean image is already a false alarm excluded. Sizes assume ~10 px/mm
(rough; the new frames are closer, so real dots are smaller than stated).

| diameter | lilac | red | yellow |
|---|---|---|---|
| 0.4 mm | 0% | 0% | 0% |
| 0.8 mm | 43% | 10% | 60% |
| 1.2 mm | 77% | 25% | 87% |
| 1.6 mm | 83% | 58% | 100% |
| 2.4 mm | 100% | 95% | 97% |

- **Red brick is the hardest**: dark/red/purple marker on saturated red has little contrast; reliable
  only from ~2.4 mm. Blue marker is the easiest on every colour.
- **Purple on lilac** (same-tone marker) is missed up to 1.6 mm – the case per-colour models
  will not fix; needs better light/focus.
- Aug3+flip raises small-dot detection on red (1.6 mm: 58% → 80%) but adds false alarms.

### Augmentation at equal false-alarm rate (decides it)

Same three FP16 models, but the threshold is set from each model's own clean-image scores, so all
allow the same number of false alarms on the 21 clean bricks. Sweep = synthetic dots above.

| model | FA | real defects | 0.8 mm | 1.2 mm | 1.6 mm | 2.4 mm |
|---|---|---|---|---|---|---|
| base | 0/21 | 16/20 | 34% | 59% | 76% | 96% |
| aug3+flip | 0/21 | 15/20 | 33% | 62% | 78% | 96% |
| aug3+flip, coreset 0.025 | 0/21 | 14/20 | 32% | 61% | 77% | 95% |
| base | 1/21 | 16/20 | 50% | 70% | 85% | 98% |
| aug3+flip | 1/21 | 16/20 | 50% | 71% | 86% | 97% |
| aug3+flip, coreset 0.025 | 1/21 | 16/20 | 51% | 70% | 85% | 96% |

**Conclusion: augmentation gives no real gain** – the earlier "more small dots caught" was only a
lower effective threshold (more false alarms). Keep `--augment` off by default (tool stays for
later, e.g. if the station has more pose/light variation). The default threshold (0.5) sits right
at the clean-score maximum (0.52) → tune the threshold / `uncertain` band on station data instead.

## 2026-11-14 — brick ROI vs raw frames, with a fake gripper in the frame (MacBook, CPU)

Data: e08b921 (30 `training/good`, 12 `test/good`, 12 `test/defective`; bricks lie on a white
table at random positions/rotations and fill only ~15–25% of the frame).
"Clutter" test set = the same 24 test images with grey/black "gripper fingers" drawn touching the
brick from both sides, plus blue tape and dark dots on the table away from the brick.
PatchCore, 3 seeds each, model's own threshold.

| model | clean AUROC | clean false alarms | clean caught | clutter AUROC | clutter false alarms | clutter caught |
|---|---|---|---|---|---|---|
| raw frames | 0.896 / 0.917 / 0.896 | 3, 1, 3 of 12 | 10/12 | **0.50** (all 3 seeds) | **12/12** | 12/12 |
| `--roi` | 0.979 / 0.986 / 0.986 | 1 of 12 | 11–12/12 | 0.986 / 0.979 / 0.972 | 1 of 12 | 10/12 |

- **Raw frames break completely** once anything new appears next to the brick: every good brick is
  "defective" (AUROC 0.5 = coin flip). This is exactly the gripper problem.
- **ROI fixes it**: the gripper/table are greyed out per frame, the clean and clutter crops look
  almost identical, and scores barely change.
- **ROI also helps on clean frames** (AUROC ~0.90 → ~0.98, more defects caught): after cutting and
  straightening, the brick fills the 256 px model input instead of ~50 px, so small dots survive.
- The 2 defects missed under clutter are tiny dots on the brick edge with scores 0.485 / 0.496,
  just under the 0.5 threshold - not caused by the gripper (same crops as clean). An `uncertain`
  band around 0.5 with re-inspection would catch them.
- Limits not tested here: gripper fingers that *cover* the inspected face, gripper of the same
  colour as the brick, real shadows of the gripper on the brick.

**Decision: use `--roi` for training and inference** (and `Scorer(..., roi=True)` in robot code).

## 2026-11-14 — Ivan's gripper frames on a cluttered background (raw 1920×1080, MacBook CPU)

Data: `dataset/raw/training/good` (311b77d), 30 full frames: brick held in the gripper (gripper held
by hand), camera fixed and tilted ~30°, busy background (wooden table, rails, blue balloon, ceiling
lights, people's faces). Colours: cream (0–6), red (7–11, 21–29), very pale lilac (12–20).
Only `good` exists yet, so: 20 frames train, 10 held out as `test/good`; 15 synthetic black marks
(dot 16 px, dot 32 px, 120 px line) on the 5 red held-out frames (red is easy to mask) as `test/defective`.
PatchCore, seed 1.

| input | AUROC | clean scores | false alarms @0.5 | caught @0.5 | caught at 0 false alarms |
|---|---|---|---|---|---|
| full frame | 0.78 | 0.18–1.00 | 5/10 | 14/15 | 0/15 |
| centre 700 crop | **1.00** | 0.00–0.17 | **0/10** | 13/15 | **15/15** |
| `--roi` (roi.py) | 0.31 | 0.64–1.00 | 10/10 | 15/15 | 0/15 |
| old white-table model, full | 0.50 | 1.00 | 10/10 | – | 0/15 |
| old white-table model, centre | 0.70 | 0.91–1.00 | 10/10 | – | 0/15 |

- **Full frame fails** because people/hands in the background change: the heatmap of a clean brick
  lights up on a face (score 1.00).
- **roi.py fails on clutter**: it found the brick in 1/10 test frames; otherwise it cut out the table
  or the blue balloon (it assumes a plain background and takes the most colourful blob).
- **Centre crop works** here because the gripper is held roughly in the image centre: clean bricks stay
  cold, the line is localised exactly. The 2 misses at 0.5 are 16 px dots (scores just under 0.5);
  with a threshold from clean frames (0.17) all 15 are caught → threshold calibration matters again.
- **A model does not transfer between setups**: the white-table model calls every gripper frame defective.
  Retrain per camera/background setup.
- Caveats: 10 test frames from the same sequences as training (optimistic), synthetic defects only on red,
  one seed. Need Ivan's real `test/good` + `test/defective` in this setup to confirm.
## 2026-11-14 — segmentation + canonical crop (Windows laptop, CPU only)

Data: commit e08b921 (30 `training/good`, 12 `test/good`, 12 `test/defective`, 3 colours).
PatchCore defaults (wide_resnet50_2, layer2+3, 256×256), seed 42, Anomalib 2.7.0.

New step `vision/segment.py`: brick mask from Lab colour distance to the paper → morphology
(open, close, largest component, fill holes) → `minAreaRect` → rotate long side horizontal,
studs on top → tight crop + 8 % margin, background flat grey.

Two problems found in the raw photos before training:
- The brick covers only 5–26 % of the frame, and `test/defective` bricks were shot much closer
  (crop up to 552 px vs ~230–350 px in training): the raw model can score "bigger brick"
  instead of the defect.
- After straightening, all 30 training bricks had studs up and all 12 defective studs down
  (same face, turned 180° in the photo). Fixed by `studs_at_bottom()`: the stud side of the
  silhouette is less filled.
- The old HSV-saturation brick mask finds nothing on lilac (falls back to the image centre);
  `synth_defects.py` now uses the new mask.

### Results (same photos, same model, only the pre-processing differs)

| data | AUROC (all test) | defective above highest good | max good | min defective |
|---|---|---|---|---|
| raw 700×700 | 0.889 | 4/12 | 0.725 | 0.379 |
| **segmented** | **1.000** | **12/12** | 0.395 | 0.463 |

Synthetic defects (72 on `test/good`, 6 per image, seed 0), AUROC per kind vs the 12 good:

| kind | raw | segmented |
|---|---|---|
| tiny (1–2 mm) | 0.685 | **1.000** |
| edge | 0.798 | **0.988** |
| dot | 0.946 | 1.000 |
| line | 0.951 | 1.000 |
| curve | 0.981 | 1.000 |
| scribble | 1.000 | 1.000 |
| all | 0.899 | **0.998** (71/72 above every good) |

Heatmaps: with segmentation the hot spot sits exactly on the marker dot/line; on raw photos the
whole brick glows and dots are weak.

### Caveats / next

- Tiny test set (12 + 12): AUROC 1.0 here means "separable", not "perfect".
- The threshold is still wrong: Anomalib's adaptive 0.5 (fit on half of `test/`) labels 3 real
  defects good (scores 0.46–0.50) although every defective scores above every good. Re-tune
  the threshold on a bigger `test/` before the run (the gap here was 0.395 → 0.463).
- Remaining warm areas on good bricks: the studs (perspective changes) and the ragged lilac mask
  edge. Options: more training poses, mask the anomaly map with an eroded brick mask, depth mask
  from the RealSense for cleaner outlines.
- `test/defective` #011 is a different face (2×2 top view) that is never in training.
- Reproduce: `python vision/segment.py --root dataset --out dataset_seg`, then
  `python vision/train.py --root dataset_seg`.

## 2026-11-14 — brick held in the gripper, cluttered background (`vision/good/good`, 30 good)

1920×1080 photos of yellow, red and white bricks held in the gripper; background = wooden table,
cables, hands, people, blue LED strips. No paper, so the paper-colour mask picks the **table** in
23/30 photos.

New `segment.py --mode scene`: mean-shift flattening → k-means (8 colours, Lab) → pick the region
that is big × convex × central, smooth, not touching the border → GrabCut refine → strong opening
(cuts glued-on bits of hand/gripper) → same morphology + canonical crop as before.
Brick found in 30/30; clean crop in 28/30 (#003, #013 came out studs-down because of stray bits;
#000 is a close-up where only the face is kept). ~0.9 s per 1920×1080 photo on a Ryzen 7 CPU.

| model | data | result |
|---|---|---|
| trained on paper photos (`dataset_seg`) | these 30 good crops | 16/30 good flagged defective: new camera, light, white colour → must retrain per setup |
| trained on 20 of these crops | 10 held-out good + 60 synthetic defects | AUROC 0.848 (0.875 without the 2 bad crops) |

Synthetic AUROC by kind (without the 2 bad crops): scribble 1.00, dot/line/curve 0.97, edge 0.74,
tiny 0.60.

Heatmaps: defects light up exactly where they are, but good bricks also light up on
**reflections of the room LED strips** on the glossy surface, which move with every pose.

Next for this setup: shield the inspection spot from room light (diffuse light box / hood around
camera + brick), fixed presentation pose, 50+ good photos per colour, real defective bricks,
RealSense depth for the mask (brick and gripper are the nearest objects).

## 2026-11-14 — segment.py v2 (artefact fixes) on Ivan's gripper frames (MacBook CPU)

Same split as the gripper experiment above (20 train / 10 held-out good of `dataset/raw`,
15 synthetic black marks on the 5 red held-out frames), PatchCore seed 1, FP16 OpenVINO.

Fixes on top of Alex's `--mode scene`: colour refine (drop mask pixels far from the brick core
colour, hull ∩ mask so edge marker strokes stay), body-only orientation (opening before
`minAreaRect`), `trim_outline` (cut thin columns/rows sticking out of the straightened box),
`mask_quality` (area, aspect, solidity, border, >8 % trimmed → `uncertain`), optional search window.

| input | AUROC | clean scores | false alarms @0.5 | caught at 0 false alarms |
|---|---|---|---|---|
| centre 700 crop | 1.00 | 0.00–0.17 | 0/10 | 15/15 |
| segment v1 (Alex, as is) | 0.95 | 0.00–0.66 | 1/10 | 9/15 |
| **segment v2** | **1.00** | 0.00–0.15 | 0/10 | **15/15** |

Both false alarms of v1 were glued-on gripper/hand bits (#2: 0.66, #20: 0.48), not the brick.
v2: clean crops on 28/30 frames (v3 below: 29/30); #20 is flagged by the quality check (would be re-inspected);
#3 still comes out studs-down; #0 (blurred close-up) keeps only the face. ~0.7 s per 1920×1080
frame on the Mac (CPU, not optimised).

**Shifted bricks** (test frames translated, models unchanged — "robustness to new positions"):

| shift | centre 700 AUROC / false alarms | segment v2 AUROC / false alarms |
|---|---|---|
| +300 px right | 0.69 / 0 (6/15 caught) | **1.00 / 0** (15/15) |
| −250, −150 px | 0.58 / **10/10** | **1.00 / 0** (15/15) |

**Decision: `--seg scene` for training and inference** (`train.py --seg scene`,
`predict.py --seg scene`, `Scorer(..., seg="scene")`); centre crop only works while the brick
happens to be in the middle. Caveats as above: held-out frames come from the same sequences,
synthetic defects only on red, one seed; needs Ivan's real `test/` frames.

### segment.py v3 — brick geometry (29/30 clean)

`box_constrain`: in the straightened mask, find the brick body (opening wider than a stud),
decide the studs side by what sticks out over the central body columns (tie → old edge-fill rule),
keep only body + stud height on that side; side bits of hand/gripper no longer tilt or flip the crop.
Result on the 30 gripper frames: 29/30 clean, all studs up (fixed #1, #3, #16, #20); #20 still
flagged by the quality check (9 % trimmed → would be re-inspected, crop itself is clean).
Left: #0 is a blurred close-up where the studs are in shadow and the mask keeps only the face.
Tried and dropped: colour region growing (did not recover the shadowed studs) and a sharpness
check (blurred #0 = 9.6 vs sharp #3 = 10.1, not separable). Depth from the RealSense would fix #0
(studs are geometry, not colour). Model metrics unchanged: AUROC 1.00, 0/10 false alarms, 15/15,
also with bricks shifted ±300 px.

### segment.py v4 — pale gripper next to a pale brick (#20)

#20 (pale lilac brick) had a strip of the light-grey gripper glued to its side: colour cannot
separate them. Added `snap_sides` (cut at the brick's own straight side edge: strongest *inner*
horizontal Lab gradient, skipping the mask boundary), a chroma filter in the stud band (keeps
studs in shadow, drops greenish/skin slivers) and smaller side slack (1.5 %). Left strip gone,
right wedge smaller but not gone; #20 stays flagged `uncertain` (11 % trimmed). Other 29 crops
unchanged. Tried and dropped: a strict colour core as the body box (shading broke it → studs cut
on 12/30 frames). Metrics unchanged: AUROC 1.00, 0/10 false alarms, 15/15, also shifted.

## 2026-11-14 — inspection decision (`inspect_brick.py`, `calibrate.py`), branch `integration`

Model: PatchCore FP16 trained on segmented gripper frames (20 train). Calibrated on the 10 held-out
clean frames + 15 synthetic defects (same frames are scored below → optimistic, real station frames
needed): low 0.12 (clean max 0.09), high 0.28 (defective min 0.43) → separable.

Per face (25 frames): 9/9 clean scored → good, 14/14 defects scored → defective, 2 → uncertain
(segmentation quality check: #20 pale gripper; one line defect on the brick edge where 11 % of the
mask got trimmed → re-inspect, not a wrong verdict). ~1.0–1.5 s per face on the Mac CPU, almost all
segmentation; the model is ~50 ms.
Per brick: all-good brick → all faces checked; defect on face 2 of 4 → stop after 2 (early exit);
doubtful face → re-shot, then `uncertain` (or `defective` with `uncertain_as="defective"`).

## 2026-11-14 — robot collection sessions in `data/` (ground / gripper / top), variant grid (MacBook CPU)

Data: 4 sessions from Alex's `collection/run.py` (red, light_purple, light_green, light_yellow),
101 captures, good only, arm posed **by hand** (free poses, fingers sometimes in view), 640×480.
Faces labelled by hand in `dataset/faces.csv` (ground view): top 25, side 25, bottom 21, no brick 21,
occluded/cut 9. Runner: `vision/experiment.py` (README §9); test = ~25 % of the arm-pose groups per
(colour, face); 6 synthetic marker defects drawn on each raw test frame (9 ground / 12 gripper frames
with a clean reference mask -> 54 / 72 defective frames). One seed, small test set: differences of
a few defects are noise.

**Cameras**
- `ground` (RealSense low, brick in the gripper): the inspection view. Depth is useless here: 5–20 %
  valid pixels, the brick is closer than the D415 minimum range.
- `gripper` (webcam in the gripper): sees the brick in **every** capture, always at the same spot
  between the jaws, and a different face than `ground` → a second view per pose for free.
- `top` (RealSense above the table): sees the brick only when the arm is NOT in front of the ground
  camera, a few dozen pixels big → not an inspection view (brick finding on the table only; its
  depth is clean, table at ~705 mm).

**Crop QA** (71 ground frames / 92 gripper frames, `results/experiments/qa1`)

| camera | finder | mask quality ok | crop ms | what the crops look like |
|---|---|---|---|---|
| ground | `bg` (median background, Alex's idea) | 51/71 | 124 | whole brick every time, sometimes a corner of the jaw |
| ground | `scene` (`segment.py`) | 44/71 | 931 | picks the white gripper jaw instead of the brick ~9×, slivers |
| ground | `bg_scene` (bg box → scene) | 42/71 | 1364 | fewer jaw picks, some slivers, one blue bowl |
| gripper | `window` (scene in the jaw area) | 33/92 | 1088 | often half a brick |
| gripper | `window_raw` / `window_tight` (fixed crop) | 92/92 | 0 | brick + jaws + changing background |

**Anomaly results** (PatchCore unless noted; `results/experiments/grid1`, `grid2`)

| camera | finder | grouping | AUROC | caught@0FA | paired |
|---|---|---|---|---|---|
| ground | full (no crop) | all | 0.39 | 1/54 | – |
| ground | bg | all | 0.62 | 13/54 | 0.80 |
| ground | bg | per_face | 0.65 | 24/54 | 0.67 |
| ground | bg | per_face_colour | 0.69 | 26/54 | 0.63 |
| ground | bg_scene | per_face_colour | 0.69 | **33/54** | 0.78 |
| ground | scene | per_face | 0.56 | 15/54 | – |
| gripper | window | all | **0.84** | 31/72 | 0.76 |
| gripper | window_tight / window_raw / full | all | 0.51–0.56 | 0–2/72 | – |

caught@0FA = defective crops above every good test crop of the same model; paired = defective crop
above the same frame without the mark (signal without pose variation).

What we learned
- **Cropping is mandatory**: full frame is worse than chance (0.39). Alex's median background is the
  best ground finder here: cleanest crops and 8× faster than `scene`. `segment.py` `scene` alone
  confuses the white jaw with a pale brick on this camera.
- **Separate models per face help** (bg: 13 → 24/54), per face + colour a bit more (but with 4–5
  training frames per model and 1–2 test goods, caught@0FA gets easier; treat as a trend).
- PatchCore beats PaDiM in every pair; CLAHE and the "on brick" score make no consistent difference.
- Heat maps put the hot spot exactly on the synthetic mark, but **good crops score as high** because
  of hand-posed perspective (tilted faces), bits of jaw/background and ~17–19 frames per face for 4
  colours. Paired 0.6–0.8: on a 640×480 frame tiny dots / dark ink on red barely change the score.
- The gripper view is promising (0.84 with `window`), but its numbers are biased toward frames where
  `window` cut the brick cleanly (only those got synthetic defects); fixed crops without segmentation
  fail because the background behind the brick changes with the arm pose.
- Alex's `anomaly/` result (AUROC 0.725, all good flagged) came from mixing faces in one model and
  same-pose frames in train and test, not from the crop.

Next (in order of expected gain)
1. Collect with the **robot's fixed inspection poses** (`docs/inspection_sequence.md`), 30+ good per
   face per colour, plus a **separate test session** and **real marker defects**.
2. Ground camera colour at **1280×720 or higher** for inspection (bricks are 150–250 px wide now).
3. Ground: `bg` crop (rebuild the background on the station) → per-face PatchCore. Gripper: a better
   brick finder inside the jaw window (the jaws never move in this camera, so a jaw mask + colour
   would do) before trusting its numbers.
4. Then export the winner (`export.py`) and measure CPU / GPU / NPU on the station (`benchmark.py`).
