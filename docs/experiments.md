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
