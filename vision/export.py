"""Export a trained checkpoint to OpenVINO IR in several precisions.

FP32 = reference, FP16 = default for GPU, INT8 (NNCF post-training quantisation, calibrated on
training/good) = smallest/fastest, usually best for NPU and CPU.

Example:
    python vision/export.py --ckpt results/Patchcore/bricks/v0/weights/lightning/model.ckpt
    # -> results/export/patchcore/{fp32,fp16,int8}/weights/openvino/model.xml
"""

import argparse
from pathlib import Path

from anomalib.deploy import CompressionType, ExportType
from anomalib.engine import Engine

from common import MODELS, make_datamodule

PRECISIONS = {"fp32": None, "fp16": CompressionType.FP16, "int8": CompressionType.INT8_PTQ}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--model", choices=MODELS, default="patchcore")
    p.add_argument("--precisions", nargs="+", choices=PRECISIONS, default=list(PRECISIONS))
    p.add_argument("--root", type=Path, default=Path("dataset"), help="dataset root (INT8 calibration)")
    p.add_argument("--out", type=Path, default=Path("results/export"))
    return p.parse_args()


def export(ckpt: Path, model_name: str, precisions, root: Path, out: Path, size: int = 256) -> dict[str, Path]:
    """Export with a fully static input [1, 3, size, size].

    Anomalib 2.7's default "dynamic batch" export traces with batch=2 and PatchCore bakes it in,
    so the IR rejects single images. A static shape fixes that and is what the NPU wants anyway.
    Callers must resize frames to size x size before inference (benchmark.py does).
    """
    engine = Engine(accelerator="cpu", logger=False)
    paths = {}
    for prec in precisions:
        model = MODELS[model_name][0]()
        # INT8 calibration runs the static IR on training/good, so feed it single 256x256 images
        dm = make_datamodule(root, batch_size=1, resize=size) if prec == "int8" else None
        if dm is not None:
            dm.setup()
        path = engine.export(
            model=model,
            export_type=ExportType.OPENVINO,
            export_root=out / model_name / prec,
            input_size=(size, size),
            compression_type=PRECISIONS[prec],
            datamodule=dm,
            ckpt_path=ckpt,
            onnx_kwargs={"dynamic_axes": {}, "dynamic_shapes": None},
        )
        paths[prec] = Path(path)
        print(f"{prec}: {path}")
    return paths


def main() -> None:
    args = parse_args()
    export(args.ckpt, args.model, args.precisions, args.root, args.out)


if __name__ == "__main__":
    main()
