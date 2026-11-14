"""Score a single image or a folder of images and save heatmap overlays.

Example:
    python vision/predict.py --ckpt results/.../model.ckpt --input dataset/test/defective --out results/predict
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore, Stfpm

MODELS = {"patchcore": Patchcore, "padim": Padim, "stfpm": Stfpm}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True, help="image file or folder")
    p.add_argument("--model", choices=MODELS, default="patchcore")
    p.add_argument("--accelerator", default="cpu")
    p.add_argument("--out", type=Path, default=Path("results/predict"))
    p.add_argument("--roi", action="store_true", help="align bricks first (model trained with --roi)")
    p.add_argument("--seg", choices=["scene", "paper"], default=None,
                   help="segment bricks first (model trained with the same --seg)")
    return p.parse_args()


def segment_input(src: Path, dst: Path, mode: str) -> Path:
    """Segment one image or a folder of raw frames into dst; prints frames that would be 'uncertain'."""
    from segment import IMAGE_EXT, process
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXT)
    dst.mkdir(parents=True, exist_ok=True)
    for p in files:
        try:
            _, crop, _, info = process(cv2.imread(str(p)), mode=mode)
        except ValueError:
            print(f"{p.name}: no brick found -> uncertain")
            continue
        if not info["ok"]:
            print(f"{p.name}: segmentation doubtful ({', '.join(info['reasons'])}) -> uncertain")
        cv2.imwrite(str(dst / f"{p.stem}.png"), crop)
    return dst


def overlay(image_path: Path, anomaly_map, out_path: Path) -> None:
    """Blend the anomaly map over the image on a FIXED 0..1 scale.

    Anomalib normalises maps so that 0.5 is the decision threshold. Per-image min-max
    stretching would paint noise on good bricks red, so we only clip to [0, 1].
    """
    img = cv2.imread(str(image_path))
    amap = np.asarray(anomaly_map, dtype=np.float32).squeeze()
    amap = cv2.resize(amap, (img.shape[1], img.shape[0]))
    amap = (255 * np.clip(amap, 0.0, 1.0)).astype(np.uint8)
    heat = cv2.applyColorMap(amap, cv2.COLORMAP_JET)
    cv2.imwrite(str(out_path), cv2.addWeighted(img, 0.6, heat, 0.4, 0))


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = args.input
    if args.seg:
        data = segment_input(args.input, args.out / "seg_input", args.seg)
    elif args.roi:
        from roi import align_tree
        data = args.out / "roi_input"
        align_tree(args.input, data)
    engine = Engine(default_root_dir=args.out, accelerator=args.accelerator)
    preds = engine.predict(model=MODELS[args.model](), data_path=data, ckpt_path=args.ckpt)

    print(f"{'image':40s} {'score':>7s}  label")
    for batch in preds:
        for i, path in enumerate(batch.image_path):
            score = float(batch.pred_score[i])
            label = "defective" if bool(batch.pred_label[i]) else "good"
            print(f"{Path(path).name:40s} {score:7.3f}  {label}")
            overlay(Path(path), batch.anomaly_map[i].cpu().numpy(), args.out / f"{Path(path).stem}_heat.png")


if __name__ == "__main__":
    main()
