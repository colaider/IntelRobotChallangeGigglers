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
