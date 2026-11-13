"""Capture inspection frames from a webcam into dataset/<split>/<class>/NNN.png.

SPACE saves a frame, Q (or ESC) quits.

Example:
    python vision/capture.py --cls good --camera 0 --width 1920 --height 1080 --crop 800
"""

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["training", "test"], default="training", help="dataset split")
    p.add_argument("--cls", choices=["good", "defective"], required=True, help="class folder to save into")
    p.add_argument("--camera", type=int, default=0, help="camera index")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--crop", type=int, default=0, help="fixed centre square crop size in px (0 = no crop)")
    p.add_argument("--out", type=Path, default=Path("dataset"), help="dataset root")
    return p.parse_args()


def centre_crop(frame, size: int):
    h, w = frame.shape[:2]
    size = min(size, h, w)
    y0, x0 = (h - size) // 2, (w - size) // 2
    return frame[y0 : y0 + size, x0 : x0 + size]


def next_index(folder: Path) -> int:
    nums = [int(f.stem) for f in folder.glob("*.png") if f.stem.isdigit()]
    return max(nums, default=-1) + 1


def main() -> None:
    args = parse_args()
    out_dir = args.out / args.split / args.cls
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    idx = next_index(out_dir)
    print(f"Saving to {out_dir} (next index {idx:03d}). SPACE = save, Q = quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame")
            break
        if args.crop:
            frame = centre_crop(frame, args.crop)

        preview = frame.copy()
        cv2.putText(preview, f"{args.cls}  next: {idx:03d}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("capture", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            path = out_dir / f"{idx:03d}.png"
            cv2.imwrite(str(path), frame)
            print(f"saved {path}")
            idx += 1
        elif key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
