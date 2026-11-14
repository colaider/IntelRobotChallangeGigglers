# Vision pipeline — what we do and what is proven

One page for any engineer joining. Details and all numbers: [`experiments.md`](experiments.md);
commands: [`../vision/README.md`](../vision/README.md).

## Goal

Inspection camera frame of a brick → **good / defective / uncertain** → the robot drops the brick
into the right bowl (or re-inspects). Defects = marker drawings on the brick. Everything runs on
the laptop with OpenVINO (CPU / GPU / NPU).

## Flow

```
 OFFLINE (build the model)                          ONLINE (every brick on the robot)

 capture.py ──► dataset/raw/  (full frames)         camera frame
                    │                                   │
                    ▼                                   ▼
 segment.py: find brick → clean mask →            segment.process()  ── mask looks wrong? ──► UNCERTAIN
   drop gripper/hand bits → straighten                  │                                    (re-inspect)
   (studs up) → crop, grey background                   ▼
                    │                              resize 256×256
                    ▼                                   │
 train.py  PatchCore (Anomalib) on good only            ▼
                    │                              OpenVINO model (NPU / GPU / CPU)
                    ▼                                   │
 export.py  OpenVINO IR  FP32 / FP16 / INT8             ▼
                    │                              score + heatmap
                    ▼                                   │
 benchmark.py / sensitivity.py                          ▼
   latency per device, accuracy, smallest       score < low  → GOOD      ─┐
   defect we catch                              score > high → DEFECTIVE ├─► instruction to the
                                                in between   → UNCERTAIN ─┘   robot policy (VLA)
```

## Steps and what each one gives us

| # | Step | Tool | What it gives |
|---|---|---|---|
| 1 | Capture full frames | `capture.py` → `dataset/raw/` | Nothing is lost; any crop can be redone later without reshooting |
| 2 | **Segment the brick** | `segment.py --mode scene` | Model sees only the brick, always straightened and studs up: background, people, gripper, position no longer matter |
| 3 | Quality check of the mask | `segment.mask_quality` | Bad masks become `uncertain` instead of a wrong verdict |
| 4 | Train on good bricks only | `train.py --seg scene` (PatchCore) | No defect examples needed; ~20 s training; a new colour can be taught in seconds |
| 5 | Export + quantise | `export.py` (OpenVINO FP32/FP16/INT8) | INT8 ~11 % faster, same accuracy; static input 256×256 (needed for NPU) |
| 6 | Measure | `benchmark.py`, `sensitivity.py`, `synth_defects.py` | Latency per device; detection rate per defect size / marker colour / brick colour |
| 7 | Decide | thresholds + `uncertain` band | Uncertain bricks are re-inspected → fewer wrong sorts |

## What is proven (MacBook CPU, our photos)

| Finding | Evidence |
|---|---|
| **Segmenting the brick is the biggest win** | Bricks in the gripper on a busy background: full frame AUROC 0.78 (a face in the background scored 1.00); segmented 1.00, 0 false alarms, 15/15 synthetic defects caught |
| **Segmentation survives a moved brick, a fixed crop does not** | Bricks shifted 300 px: centre crop AUROC 0.58–0.69 (10/10 false alarms); segmented 1.00 |
| Cutting off glued-on gripper bits matters | Raw segmentation: 2 false alarms, both caused by gripper bits; after the fix: 0. Clean crops 24/30 → 29/30 |
| Straight, tight crop catches smaller defects | Plain background (Alex): AUROC 0.89 → 1.00; tiny 1–2 mm dots 0.69 → 1.00 |
| PatchCore is the best model here | PatchCore AUROC 0.83–0.94 vs PaDiM 0.79, STFPM 0.62 |
| INT8 is free speed | 49 vs 55 ms on CPU, same AUROC; size barely changes (memory bank dominates) |
| A model does not transfer between setups | Table-trained model flags every gripper frame → retrain per camera/background/light |

## What did NOT help (so do not spend time on it)

| Tried | Result |
|---|---|
| Augmenting good images (rotate/shift/light) | Same detection at equal false-alarm rate; bigger and slower model |
| Median filter before the model | No change (frames are downscaled to 256 px anyway) |
| `roi.py` (older cut-out) | Works on plain paper only; on clutter it cut out the table/balloon (1/10 right) |
| Full frame without cropping | Learns the background; anything new in the room = "defect" |

## Known limits / next

- Thresholds: the default 0.5 is not calibrated → set low/high from clean frames of the real setup.
- Hardest cases: purple marker on lilac, dark marker on red, dots < 1 mm.
- Glossy plastic reflects room lights → diffuse light / hood at the inspection spot.
- Numbers above come from small sets (10–20 test frames, partly synthetic defects) → re-check on
  the station with real defective bricks; measure GPU/NPU latency there.
- On the station: use the arm's fixed inspection pose (`--window`) and RealSense depth for the mask.
