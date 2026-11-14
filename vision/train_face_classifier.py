"""Face classifier per camera: brick crop -> top / bottom / side. OPTIONAL (inspector mode "classifier").

Why: with an unknown brick orientation the face a camera sees is not known from the arm pose. The
default is model competition (each per-face PatchCore scores the crop, the lowest score wins); this
small network is the alternative / cross-check, and a second OpenVINO model that can run on another
device than Anomalib (e.g. classifier on GPU, PatchCore on NPU).

Data = the same per-face datasets (folder = label): dataset/views/<camera>_<face>/
    training/good                -> training
    test/good + test/defective   -> evaluation (defective crops must still be recognised as their face)
Model: torchvision MobileNetV3-Small pretrained on ImageNet, last layers fine-tuned, input 224.
Export: OpenVINO IR (FP16) + classes.json; registered in results/models/models.json -> cameras.<cam>.classifier

Example:
    python vision/train_face_classifier.py --cameras gripper front
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

FACES = ["top", "bottom", "side"]
SIZE = 224
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--views", type=Path, default=Path("dataset/views"))
    p.add_argument("--out", type=Path, default=Path("results/models"))
    p.add_argument("--cameras", nargs="+", default=["gripper", "front"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--min-per-class", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def prep(bgr: np.ndarray) -> np.ndarray:
    """BGR crop -> normalised CHW float32 (same as at inference)."""
    rgb = cv2.cvtColor(cv2.resize(bgr, (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return ((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)


def augment(bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-15, 15), rng.uniform(0.9, 1.1))
    out = cv2.warpAffine(bgr, m, (w, h), borderValue=(128, 128, 128))
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    return np.clip(out.astype(np.float32) * rng.uniform(0.85, 1.15), 0, 255).astype(np.uint8)


def load(root: Path, subs: list[str], classes: list[str]):
    items = []
    for ci, f in enumerate(classes):
        for sub in subs:
            items += [(cv2.imread(str(p)), ci) for p in sorted((root.parent / f"{root.name}_{f}" / sub).glob("*.png"))]
    return items


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    reg_path = args.out / "models.json"
    registry = json.loads(reg_path.read_text()) if reg_path.exists() else {"cameras": {}}
    for cam in args.cameras:
        base = args.views / cam
        counts = {f: len(list((args.views / f"{cam}_{f}" / "training/good").glob("*.png"))) for f in FACES}
        classes = [f for f in FACES if counts[f] >= args.min_per_class]
        print(f"\n=== {cam}: training images per face {counts}")
        if len(classes) < 2:
            print(f"  need >= 2 faces with >= {args.min_per_class} images, skipped")
            continue
        train = load(base, ["training/good"], classes)
        test = load(base, ["test/good", "test/defective"], classes)

        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
        for p in model.features[:-3].parameters():  # fine-tune only the last blocks + head
            p.requires_grad = False
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        lossf = nn.CrossEntropyLoss()
        for ep in range(args.epochs):
            model.train()
            rng.shuffle(train)
            tot = 0.0
            for i in range(0, len(train), 16):
                batch = train[i:i + 16]
                x = torch.from_numpy(np.stack([prep(augment(im, rng)) for im, _ in batch]))
                y = torch.tensor([c for _, c in batch])
                opt.zero_grad()
                loss = lossf(model(x), y)
                loss.backward()
                opt.step()
                tot += float(loss.detach()) * len(batch)
            if ep % 5 == 4 or ep == args.epochs - 1:
                print(f"  epoch {ep + 1}: loss {tot / len(train):.3f}")
        model.eval()
        acc = None
        if test:
            with torch.no_grad():
                pred = model(torch.from_numpy(np.stack([prep(im) for im, _ in test]))).argmax(1).numpy()
            acc = float((pred == np.array([c for _, c in test])).mean())
            print(f"  test accuracy {acc:.2f} on {len(test)} crops (clean + defective)")

        out = args.out / f"{cam}_face_classifier"
        out.mkdir(parents=True, exist_ok=True)
        ovm = ov.convert_model(model, example_input=torch.zeros(1, 3, SIZE, SIZE))
        xml = out / "model.xml"
        ov.save_model(ovm, str(xml), compress_to_fp16=True)
        meta = {"xml": str(xml), "classes": classes, "size": SIZE, "mean": MEAN.tolist(), "std": STD.tolist(),
                "test_accuracy": acc, "n_train": len(train)}
        (out / "classes.json").write_text(json.dumps(meta, indent=2))
        registry.setdefault("cameras", {}).setdefault(cam, {})["classifier"] = meta
        print(f"  saved {xml}")
    args.out.mkdir(parents=True, exist_ok=True)
    reg_path.write_text(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()
