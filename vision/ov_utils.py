"""OpenVINO compile helpers shared by the inspector and the station check.

- device fallback: NPU -> HETERO:NPU,CPU -> GPU -> CPU (GPU -> CPU), each candidate is compiled AND
  run once on a dummy input, so unsupported operators fail here and not in the middle of a demo
- model cache (CACHE_DIR): the first NPU/GPU compile can take seconds; later starts load the blob
- PERFORMANCE_HINT=LATENCY: one frame at a time (robot loop), not batch throughput

    from ov_utils import compile_with_fallback
    r = compile_with_fallback("model.xml", "NPU")
    r.compiled, r.device, r.compile_s, r.tried     # tried = [(device, error or "ok"), ...]
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

CACHE_DIR = Path("results/ov_cache")
_core = None


def resize_for_model(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize a crop to the model input the way training did (torchvision bilinear Resize).

    INTER_AREA only when shrinking: when enlarging it is close to nearest-neighbour, so a small crop
    (e.g. a 76 px side face -> 256) turns blocky, PatchCore sees the blocks as anomalies and clean
    frames saturate at score 1.0 (side model: AUROC 0.64 -> 0.97 on the same test set after this fix).
    """
    if img.shape[0] == h and img.shape[1] == w:
        return img
    shrink = img.shape[0] >= h and img.shape[1] >= w
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)


def core() -> ov.Core:
    global _core
    if _core is None:
        _core = ov.Core()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _core.set_property({"CACHE_DIR": str(CACHE_DIR)})
    return _core


def chain(device: str) -> list[str]:
    """Candidates tried in order for a requested device."""
    device = device.upper()
    if device == "NPU":
        return ["NPU", "HETERO:NPU,CPU", "GPU", "CPU"]
    if device == "GPU":
        return ["GPU", "CPU"]
    return [device] if device in ("CPU", "AUTO") else [device, "CPU"]


def available() -> list[str]:
    return core().available_devices


def is_available(dev: str) -> bool:
    bases = {a.split(".")[0] for a in available()}
    if dev == "AUTO":
        return True
    if dev.startswith("HETERO:"):
        return all(p in bases for p in dev.split(":", 1)[1].split(","))
    return dev.split(".")[0] in bases


def device_name(dev: str) -> str:
    try:
        return str(core().get_property(dev, "FULL_DEVICE_NAME"))
    except Exception:  # noqa: BLE001
        return dev


def dummy_inputs(model) -> dict:
    out = {}
    for inp in model.inputs:
        ps = inp.get_partial_shape()
        shape = [d.get_length() if d.is_static else 1 for d in ps]
        if len(shape) == 4 and not ps[2].is_static:  # dynamic H/W -> 256
            shape[2] = shape[3] = 256
        et = inp.get_element_type()
        dtype = np.uint8 if et == ov.Type.u8 else np.float32
        out[inp.get_any_name()] = np.zeros(shape, dtype)
    return out


@dataclass
class Compiled:
    compiled: object = None
    device: str = ""
    compile_s: float = 0.0
    tried: list = field(default_factory=list)


def compile_with_fallback(xml, device: str = "AUTO", hint: str = "LATENCY", fallback: bool = True) -> Compiled:
    model = core().read_model(str(xml))
    res = Compiled()
    for dev in (chain(device) if fallback else [device]):
        if not is_available(dev):
            res.tried.append((dev, "not available"))
            continue
        for cfg in ({"PERFORMANCE_HINT": hint}, {}):
            try:
                t = time.perf_counter()
                c = core().compile_model(model, dev, cfg)
                c(dummy_inputs(model))  # unsupported ops / runtime errors show up here
                res.compiled, res.device, res.compile_s = c, dev, time.perf_counter() - t
                res.tried.append((dev, "ok"))
                return res
            except Exception as e:  # noqa: BLE001 - report and try the next candidate
                err = str(e).strip().splitlines()[0][:160] if str(e).strip() else type(e).__name__
                if cfg:  # retry without the hint once (HETERO / some plugins reject it)
                    continue
                res.tried.append((dev, err))
    return res
