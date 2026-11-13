# Dataset layout (committed, shared with the team)

Photos from the inspection booth: same camera, light, matte background, distance.
One sub-folder per (face type × colour) later; for now a flat layout:

```
dataset/
  training/good/     # only perfect bricks; used to train (PatchCore etc.)
  test/good/         # perfect bricks NOT in training; threshold tuning
  test/defective/    # defective bricks (e.g. marker drawings); validation only
```

File names: `<date>_<time>_c<crop>_<NNN>.png` (set by capture.py, unique across machines). Do not mix the same photo into training and test.
Capture with: `python vision/capture.py --split training --cls good --crop 700` (see vision/README.md)
