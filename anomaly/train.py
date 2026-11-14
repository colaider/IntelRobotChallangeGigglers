"""Step 2: train PatchCore on GOOD bricks only, export to OpenVINO, save everything in model/.

    .venv\\Scripts\\python.exe anomaly/train.py

model/ afterwards (this folder IS the model - copy it to the robot laptop as a whole):
    openvino/model.xml + .bin   PatchCore exported for OpenVINO (FP16, static 1x3x256x256)
    patchcore.ckpt              the Lightning checkpoint (for retraining / torch heatmaps)
    background.png              ground-camera background for the brick finder
    model_info.json             threshold, image size, colours, training counts, test metrics

Anomalib uses half of test/ (good + synthetic defects) to pick the threshold; scores are
normalised so that 0.5 = that threshold.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="wide_resnet50_2")
    ap.add_argument("--coreset", type=float, default=0.1, help="share of patches kept in the memory bank")
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    args = ap.parse_args()

    from anomalib.data import Folder
    from anomalib.deploy import CompressionType, ExportType
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
    from lightning.pytorch import seed_everything

    if not (C.DATASET / "training/good").exists():
        raise SystemExit("No dataset - run anomaly/prepare.py first.")
    seed_everything(C.SEED, workers=True)
    dm = Folder(name="bricks", root=C.DATASET, normal_dir="training/good", normal_test_dir="test/good",
                abnormal_dir="test/defective", train_batch_size=8, eval_batch_size=8, num_workers=0, seed=C.SEED)

    def make_model():
        return Patchcore(backbone=args.backbone, coreset_sampling_ratio=args.coreset,
                         pre_processor=Patchcore.configure_pre_processor(image_size=(C.IMAGE_SIZE, C.IMAGE_SIZE)))

    model = make_model()
    engine = Engine(default_root_dir=C.WORK / "runs", accelerator="cpu", logger=False)
    t0 = time.perf_counter()
    engine.fit(model=model, datamodule=dm)
    train_s = time.perf_counter() - t0
    metrics = {k: round(float(v), 4) for k, v in engine.test(model=model, datamodule=dm, verbose=False)[0].items()}
    ckpt = Path(engine.trainer.checkpoint_callback.best_model_path)

    # fresh model folder; export with a STATIC input (what the NPU needs; dynamic batch export
    # bakes in batch=2 for PatchCore in Anomalib 2.7)
    if C.MODEL.exists():
        shutil.rmtree(C.MODEL)
    C.MODEL.mkdir(parents=True)
    xml = engine.export(model=make_model(), export_type=ExportType.OPENVINO, export_root=C.MODEL / "export",
                        input_size=(C.IMAGE_SIZE, C.IMAGE_SIZE),
                        compression_type=CompressionType.FP16 if args.precision == "fp16" else None,
                        ckpt_path=ckpt, onnx_kwargs={"dynamic_axes": {}, "dynamic_shapes": None})
    ov_dir = C.MODEL / "openvino"
    shutil.copytree(Path(xml).parent, ov_dir)
    shutil.rmtree(C.MODEL / "export", ignore_errors=True)
    shutil.copy2(ckpt, C.MODEL / "patchcore.ckpt")
    shutil.copy2(C.WORK / "background.png", C.MODEL / "background.png")

    n = {d: len(list((C.DATASET / d).glob("*.png"))) for d in ("training/good", "test/good", "test/defective")}
    colours = sorted({p.name.split("__")[0] for p in (C.DATASET / "training/good").glob("*.png")})
    info = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "PatchCore", "backbone": args.backbone, "coreset_sampling_ratio": args.coreset,
        "precision": args.precision, "image_size": C.IMAGE_SIZE, "camera": C.CAMERA,
        "threshold": 0.5,  # normalised score; edit to make the inspector stricter (lower) or looser
        "colours": colours, "counts": n, "train_seconds": round(train_s, 1),
        "anomalib_test_metrics": metrics,
        "openvino_xml": "openvino/model.xml", "background": "background.png",
    }
    (C.MODEL / "model_info.json").write_text(json.dumps(info, indent=2))
    print(f"\ntrained in {train_s:.0f}s on {n['training/good']} good bricks ({', '.join(colours)})")
    print(f"Anomalib test metrics: {metrics}")
    print(f"model saved: {C.MODEL}")


if __name__ == "__main__":
    main()
