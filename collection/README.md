# Collection: build the anomaly dataset with the real robot

Pose the SO-101 by hand (motors off), press a key, and every camera frame, the RealSense depth
and the arm's joint angles are saved together. Nothing is cropped or enhanced while collecting:
`post.py` does that later from the raw files, so settings can change without collecting again.

```
collection/
  run.py       the collector window (start this)
  cameras.py   RealSense colour + depth aligned to colour, extra USB cameras, exposure control
  robot.py     SO-101 pose reader: torque off, read joints, hold/release
  session.py   writes one session folder: images, depth, meta JSON, manifest.csv
  post.py      raw sessions -> Anomalib folders (optional crop via vision/segment.py, CLAHE)
```

## Setup (once)

Use the **system Python**, not the Anomalib `.venv` (that one has a window-less OpenCV):

```bash
py -m pip install -r collection/requirements.txt
```

Only one program can use the COM port and each camera: close Physical AI Studio and
`so101_jog.py` first. Plug the RealSense cameras into USB 3 (blue) ports, ideally on different
sides of the laptop (two depth cameras + a webcam on one USB controller can run out of bandwidth).

## Our cameras

| name | device | what it sees | used for |
|---|---|---|---|
| `ground` | RealSense, on the table looking up/at the gripper | brick held by the gripper | **inspection model** (anomaly) |
| `top` | RealSense, above the table | table, bricks, bowls | brick positions, VLA |
| `gripper` | USB webcam in the gripper | close-up while grasping | VLA, extra inspection view |

Find out which serial / index is which once, then always use the same names:

```bash
py collection/run.py --list
```

(Unplug one RealSense to see which serial disappears; try `--usb test=N --no-realsense --no-robot`
for each index to find the gripper camera. RealSense cameras also show up as USB indexes.)

**Recommended camera settings (team decision, 14 Nov):** the front inspection camera at full
colour resolution without depth (depth does not work that close anyway), the gripper webcam at
1080p in MJPEG, the top camera small with depth (pick + VLA):

```bash
py collection/run.py --rs ground=<serial> top=<serial> --usb gripper=<index> \
    --cam ground=1920x1080@15:nodepth gripper=1920x1080@30 top=640x480@30 --name red_good_a
```

`--cam NAME=WxH[@FPS][:nodepth]` overrides `--size` per camera. At start the collector runs a
3 s **camera check** (`--check-seconds`, `--check-only` to just test): frames each RealSense
actually produces vs asked, USB type, webcam format. Warnings there (USB 2, fps far below the
asked rate, 1080p without MJPEG) mean missing frames / freezes later: move a camera to another
USB controller, lower the fps first (stills do not need 30 fps), then the resolution.
Never mix resolutions within one model's data.

**Resolution: collect at what you will run at.** Everything defaults to 640x480. On a RealSense,
640x480 is 4:3 and sees a narrower field than 1280x720 (16:9), so a model trained on another
mode sees different pictures. 640x480 is enough here: on our gripper photos the brick is
~240 px wide at 640x480, and the model shrinks every crop to 256 px anyway.

## Collect

```bash
py collection/run.py --rs ground=<serial> top=<serial> --usb gripper=<index> --name red_good_a --label good
```

| key | action |
|---|---|
| move the arm by hand | motors are off (**hold the arm when the collector starts**) |
| `h` | hold the arm in this pose / limp again: pose it, press `h`, take your hands out |
| `SPACE` | capture: every camera + depth + joint angles |
| `g` / `d` | label good / defective, then `1`-`9` picks the defect type (`--defects dot line ...`) |
| `a` | auto exposure + auto white balance on/off, all cameras (USB webcam: best effort) |
| `[` / `]` | exposure down/up on the RealSense cameras (manual mode) |
| `BACKSPACE` | delete the last capture |
| `q` / `ESC` | quit (arm goes limp: hold it) |

The window shows every camera in a row and each RealSense depth below it. Checks after every
capture, per camera: overexposed (>1 % clipped pixels), low sharpness (`--min-sharpness`),
auto exposure still on.

Other options: `--no-robot`, `--no-realsense`, `--port COM5`, `--size 640x480`,
`--rs-depth 640x480`, `--manual-exposure` (start with auto off), `--color red` (optional).
Without `--rs`, every connected RealSense is used and named `rs1`, `rs2` (by serial).

## How to collect well

1. Set up the light first (shield the room LEDs: they made most false alarms so far), then
   press `a` once so exposure and white balance stay fixed for the whole session.
2. **One session = one sitting.** Start a new session when the light, camera or brick set changes.
3. Shoot good bricks in several sessions; keep **at least one good session only for testing**.
4. Show bricks the way the robot will show them in the real run (same pose, same distance).
5. Aim for 50+ good per colour, and real defective bricks of every type for testing.

## Session folder

```
data/20261115_093012_red_good_a/
  session.json              cameras (model, serial, colour intrinsics, depth_scale), robot, notes, code commit
  manifest.csv              one row per capture: label, defect, colour, quality, joint degrees, files
  0001_ground_color.png     raw colour, one file per camera name
  0001_ground_depth.png     uint16 depth aligned to colour: metres = value * depth_scale
  0001_top_color.png  0001_top_depth.png
  0001_gripper_color.png    USB camera (no depth)
  0001_meta.json            pose (ticks + degrees per joint), per-camera exposure, timestamps, quality
```

Joint degrees are raw `(ticks - 2048) * 360 / 4096`, not Physical AI Studio's calibrated
angles; ticks are saved too. `data/` is gitignored: share sessions as zip / drive, not git.

## Post-process -> train

```bash
py collection/post.py --sessions data/20261115_* --test-sessions data/20261115_140000_red_test --camera ground --crop scene --out dataset_robot
py collection/post.py --sessions data/20261115_* --test-sessions data/20261115_140000_red_test --camera ground --crop scene --clahe --out dataset_robot_clahe
.venv\Scripts\python.exe vision/train.py --root dataset_robot
```

`--crop scene` (busy background, brick in the gripper) or `paper` (plain sheet) uses
`vision/segment.py`. CLAHE evens out light and makes dark marks stand out, but also boosts
noise and reflections: train with and without and keep what scores better.

**Use the current model as a filter, not a labeller:** run it over a new session and look first
at the good captures it scores high (bad light, new reflections, unusual poses) and the
defective ones it scores low (defects it misses). Fix the setup or collect more of those.
