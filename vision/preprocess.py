"""Build a model-ready image folder from originals while preserving relative paths."""

import argparse
from pathlib import Path

import cv2

from capture import crop_box

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--out", type=Path, default=Path("dataset/processed"), help="new output folder")
    parser.add_argument("--crop", type=int, default=700, help="centre square size, 0 = full frame")
    parser.add_argument("--median", type=int, choices=(0, 3, 5), default=0,
                        help="median filter kernel size, 0 = off")
    args = parser.parse_args()

    source = args.input.resolve()
    output = args.out.resolve()
    if not source.is_dir():
        parser.error(f"input directory does not exist: {source}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        parser.error("input and output directories must be separate")
    if output.exists():
        parser.error(f"use a new output directory: {output}")
    if args.crop < 0:
        parser.error("--crop must be nonnegative")

    images = sorted(path for path in source.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        parser.error(f"no images found in {source}")

    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"Could not read {path}")
        x0, y0, x1, y1 = crop_box(*image.shape[:2], args.crop)
        image = image[y0:y1, x0:x1]
        if args.median:
            image = cv2.medianBlur(image, args.median)

        destination = (output / path.relative_to(source)).with_suffix(".png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), image):
            raise SystemExit(f"Could not write {destination}")

    print(f"Processed {len(images)} images into {output}")


if __name__ == "__main__":
    main()
