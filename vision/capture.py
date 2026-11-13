"""Capture inspection frames from a webcam into dataset/<split>/<class>/.

The preview shows the full camera frame with a green box: only the area inside the box
is saved. Put the brick so that it fills most of the box. The models resize every image
to 256x256, so a square crop tight around the brick keeps the marker lines visible and
avoids squashing a 16:9 frame.

Files are named <YYYYmmdd_HHMMSS>_<NNN>.png, so photos taken by different people on
different machines never collide when merged in git.

SPACE saves a frame, Q (or ESC) quits.

Example:
    python vision/capture.py --split training --cls good --crop 700
"""

import argparse
import time
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["training", "test"], default="training", help="dataset split")
    p.add_argument("--cls", choices=["good", "defective"], required=True, help="class folder to save into")
    p.add_argument("--camera", type=int, default=0, help="camera index")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--crop", type=int, default=-1,
                   help="centre square crop size in px; -1 = largest centred square (default), 0 = full frame")
    p.add_argument("--out", type=Path, default=Path("dataset"), help="dataset root")
    return p.parse_args()


def crop_box(h: int, w: int, crop: int) -> tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) of the saved region."""
    if crop == 0:
        return 0, 0, w, h
    size = min(h, w) if crop < 0 else min(crop, h, w)
    x0, y0 = (w - size) // 2, (h - size) // 2
    return x0, y0, x0 + size, y0 + size


def main() -> None:
    args = parse_args()
    out_dir = args.out / args.split / args.cls
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    session = time.strftime("%Y%m%d_%H%M%S")
    saved = 0
    actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(f"Camera {args.camera}: {actual[0]}x{actual[1]} (asked {args.width}x{args.height})")
    print(f"Saving to {out_dir}. SPACE = save, Q = quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame")
            break
        x0, y0, x1, y1 = crop_box(*frame.shape[:2], args.crop)

        preview = frame.copy()
        cv2.rectangle(preview, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 2)
        cv2.putText(preview, f"{args.split}/{args.cls}  saved: {saved}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("capture", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            path = out_dir / f"{session}_{saved:03d}.png"
            cv2.imwrite(str(path), frame[y0:y1, x0:x1])
            print(f"saved {path}")
            saved += 1
        elif key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
