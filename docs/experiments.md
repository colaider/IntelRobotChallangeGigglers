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
