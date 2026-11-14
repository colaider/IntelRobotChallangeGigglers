# Inspection sequence for a 2×2 Duplo brick on the SO-101

How the arm shows every face of the brick to the cameras. Based on the SO-101 URDF
(`TheRobotStudio/SO-ARM100`, `Simulation/SO101/so101_new_calib.urdf`). Still to be checked on the
real arm: wrist-roll cable range, gripper opening, camera placement.

## Facts the plan is built on

| | Value | Consequence |
|---|---|---|
| Brick | Duplo 2×2: 32 × 32 × 19 mm + studs 4.5 mm; 4 identical side faces S1–S4, top T, bottom Bt (one tube) | Grasp is always across 32 mm, on two **opposite** side faces |
| Jaw vs side face | Jaws are about as wide as the 32 mm face | **The two gripped faces are fully hidden** while held — nothing of them is visible |
| Gripper | One fixed jaw (part of the gripper body) + one moving jaw on a pivot (URDF −10°…+100°) | Not a parallel gripper: close slowly, centre the brick between the jaws |
| Wrist roll | −157° … +163° (320°, **not** 360°) | Any two rolls 180° apart fit; a 90° regrasp roll always fits one way |
| Wrist flex | ±95° | Gripper points straight down only up to ~70 mm above the table |
| Elbow | ±96.8° | Gripper horizontal (to show Bt) only further out, ~380 mm from the base |
| Lengths | upper arm 116 mm, forearm 135 mm, wrist-flex axis → fingertip ≈ 160 mm, shoulder axis ≈ 117 mm above the base | Poses below are checked against these |

## Principle

**Each grasp shows everything that is not under the jaws. The regrasp is the same brick, same
place, taken by the other pair of faces (wrist rolled 90°) — the brick is never tipped over.**

## Sequence (6 frames, 2 grasps)

| # | Action | Arm (TCP x, z in mm from the base; wrist) | Frame |
|---|---|---|---|
| 0 | Brick on the table | arm in ready pose, out of the top camera's view | **T** — top camera |
| 1 | Grasp 1 from above on S1 + S3 | TCP (230, 10), gripper down, roll = g1 (from brick yaw) | — |
| 2 | Lift to the inspection pose, roll so the jaws are perpendicular to the camera axis | TCP (215, 65), gripper down, roll r | **S2** — inspection camera |
| 3 | Roll 180° | same, roll r + 180° (fits: 180° < 320°) | **S4** |
| 4 | Reach out, wrist flex to horizontal, gripper points at the camera | TCP (380, 65), gripper horizontal | **Bt** (jaw tips sit outside the face) |
| 5 | Put the brick back flat, same spot, open | TCP (230, 10) | — |
| 6 | Grasp 2 from above on S2 + S4: approach with roll g1 ± 90° (pick the sign that stays in −157…163) | TCP (230, 10) | — |
| 7 | Lift, roll r | TCP (215, 65), roll r | **S1** |
| 8 | Roll 180° | roll r + 180° | **S3** |
| 9 | Decide on the worst face score → good / defective / uncertain → bowl (shoulder pan) | | — |

Coverage: T, Bt, S1–S4 each **fully and squarely** once (no skewed views, no partial faces).
Roll r is fixed by where the inspection camera stands; with the camera in the arm's plane it is
the roll that puts the jaw axis across the camera axis, the same value in both grasps.

Uncertain face → repeat that frame at roll ±10–15° (a reflection moves, a marker does not).

Optional shortcut: if all four frames of grasp 1 (T, S2, S4, Bt) are clearly good and the rules
allow it, grasp 2 can be skipped — but S1/S3 are then never seen, so only as a speed mode.

## Rough cycle time

pick 2 s · present + 2 rolls 3 s · flex 2 s · place 2 s · regrasp 2 s · present + roll 2 s ·
bowl 2 s → **~15 s per brick** plus inference (6 frames × segmentation ~1 s on CPU today → the
main optimisation target: search window from the fixed pose, NPU for the model).

## Implications for vision

- Six frame types, three model "views": top (T), side (S1–S4, all identical faces → one model),
  bottom (Bt). Train on frames taken in exactly these poses.
- No jaw occlusion on any captured face → no masking of jaw regions needed; the gripper only
  appears at the edges (cut by `segment.py`).
- Fixed poses → the brick is always in the same place in the frame → `segment.py --window`.

## To verify on the station (before teaching the policy)

1. Wrist roll really covers r and r + 180° with the cable attached.
2. Gripper opens wide enough for 32 mm with margin; brick does not twist when the moving jaw closes.
3. Inspection camera placement sees both TCP (215, 65) and (380, 65), or move the Bt pose.
4. Placing back flat and regrasping at roll ± 90° is repeatable (brick does not slide).

---

## Appendix: spec for the browser animation (for whoever implements it)

Single HTML widget, side view, scale 1 mm = 1 px, table at the bottom, base at the left.

- **Arm**: 2-link IK, elbow up. Shoulder axis at (35, 117), L1 = 116, L2 = 135, wrist-flex axis →
  TCP = 160. Show live joint angles (shoulder, elbow, wrist flex, wrist roll) with the limits above;
  warn colour within 6° of a limit, danger colour beyond it.
- **Brick**: 32 × 32 × 19 + 4 studs (9.6 mm Ø, at ±8 mm). Projected width in the side view is
  constant 32 mm (square), so show roll by which face label faces the camera, not by width.
- **Gripper**: grey body; fixed jaw on one side, moving jaw rotating about a pivot ~72 mm behind the
  TCP (opens ~24° when releasing). Jaw width = 32 mm (covers the gripped faces completely).
  Draw the jaw nearer to the viewer over the brick, the far one behind it.
- **Cameras**: top camera above the pick spot, inspection camera on the right at z = 65.
  Flash a translucent view cone (purple top / teal inspection) when a frame is taken.
- **Keyframes**: steps 0–9 of the table above; TCP and gripper direction from the table; animate
  ~1.1 s per step with easing; play / previous / next buttons and step dots.
- **Camera panel** under the scene: three rows — "before grasp" (T), "grasp 1" (S2, S4, Bt),
  "grasp 2" (S1, S3). Each captured face drawn to scale (side 32 × 19 with 2 studs on top,
  T 32 × 32 with 4 studs, Bt 32 × 32 with one tube). Under each frame: which jaws hold which
  faces in that grasp ("held: S1, S3").
- **Coverage line**: "seen: T ✓ Bt ✓ S1 ✓ S2 ✓ S3 ✓ S4 ✓" filling in as frames are taken.
- No sliders needed (jaw width = face width, so held faces are always fully hidden).
