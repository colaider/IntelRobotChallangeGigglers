# Dataset layout (committed, shared with the team)

Photos from the inspection booth: same camera, light, matte background, distance.
One sub-folder per (face type × colour) later; for now a flat layout:

```
dataset/
  training/good/     # only perfect bricks; used to train (PatchCore etc.)
  test/good/         # perfect bricks NOT in training; threshold tuning
  test/marker/       # bricks with marker drawings; validation only
```

File names: `NNN.png`. Do not mix the same photo into training and test.
Capture with: `python vision/capture.py --split training --cls good`
