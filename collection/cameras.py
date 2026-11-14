"""Camera grabbers: Intel RealSense (colour + depth aligned to colour) and plain USB cameras.

RealSense D400 cameras are NOT LiDAR: depth comes from two infrared cameras (+ an IR dot
projector). What we save per capture:
    color  uint8 BGR, full resolution
    depth  uint16 raw units, ALIGNED to the colour image (same size, pixel (u,v) matches),
           depth_m = depth * depth_scale (depth_scale ~0.001 -> millimetres)
and once per session the colour intrinsics + depth scale (needed to turn pixels into 3-D).

Exposure: auto exposure / auto white balance change brightness and colour from shot to shot,
which an anomaly model sees as "different". Collect with them OFF (fixed values) once the
light is set up; toggle_auto() / adjust_exposure() do that. The values in force at each
capture are stored in its metadata.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Frame:
    color: np.ndarray                      # HxWx3 uint8 BGR
    depth: np.ndarray | None = None        # HxW uint16, aligned to color (RealSense only)
    meta: dict = field(default_factory=dict)


def list_realsense() -> list[dict]:
    """Connected RealSense cameras: [{"serial", "model", "usb"}] sorted by serial (stable order)."""
    import pyrealsense2 as rs

    out = []
    for d in rs.context().query_devices():
        usb = d.get_info(rs.camera_info.usb_type_descriptor) if d.supports(rs.camera_info.usb_type_descriptor) else "?"
        out.append({"serial": d.get_info(rs.camera_info.serial_number), "model": d.get_info(rs.camera_info.name),
                    "usb": usb})
    return sorted(out, key=lambda x: x["serial"])


def list_usb(max_index: int = 6) -> list[int]:
    """OpenCV camera indexes that open. RealSense cameras also appear here (as several indexes)."""
    found = []
    try:  # probing missing indexes makes OpenCV print pages of harmless warnings
        old = cv2.utils.logging.getLogLevel()
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except AttributeError:
        old = None
    try:
        for i in range(max_index):
            for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
                cap = cv2.VideoCapture(i, backend)
                ok = cap.isOpened() and cap.read()[0]
                cap.release()
                if ok:
                    found.append(i)
                    break
    finally:
        if old is not None:
            cv2.utils.logging.setLogLevel(old)
    return found


class RealSenseCamera:
    """Colour + aligned depth from one RealSense. Usage: cam = RealSenseCamera("top", serial).start()."""

    def __init__(self, name: str = "realsense", serial: str | None = None, color=(640, 480), depth=(640, 480),
                 fps=30, warmup: int = 30, depth_on: bool = True):
        """depth_on=False: colour only (e.g. the front inspection camera, where the brick is closer
        than the D415 minimum range anyway) - frees USB bandwidth for a higher colour resolution."""
        self.color_size, self.depth_size, self.fps, self.depth_on = color, depth, fps, depth_on
        self.serial, self.warmup = serial, warmup
        self.name = name
        self._pipe = self._align = self._color_sensor = None
        self.info: dict = {}

    def start(self) -> "RealSenseCamera":
        import pyrealsense2 as rs  # pip install pyrealsense2

        self._rs = rs
        if len(rs.context().query_devices()) == 0:
            raise RuntimeError("No RealSense found (USB 3 port? cable? closed in other apps?)")
        cfg = rs.config()
        if self.serial:
            cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, *self.color_size, rs.format.bgr8, self.fps)
        if self.depth_on:
            cfg.enable_stream(rs.stream.depth, *self.depth_size, rs.format.z16, self.fps)
        self._pipe = rs.pipeline()
        profile = self._pipe.start(cfg)
        self._align = rs.align(rs.stream.color) if self.depth_on else None
        dev = profile.get_device()
        depth_sensor = dev.first_depth_sensor()
        try:
            self._color_sensor = dev.first_color_sensor()
        except RuntimeError:  # e.g. D405: colour comes from the stereo module
            self._color_sensor = depth_sensor
        for _ in range(self.warmup):  # let auto exposure settle
            self._pipe.wait_for_frames()
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.info = {
            "model": dev.get_info(rs.camera_info.name),
            "serial": dev.get_info(rs.camera_info.serial_number),
            "firmware": dev.get_info(rs.camera_info.firmware_version),
            "depth_scale": depth_sensor.get_depth_scale(),
            "depth_aligned_to": "color",
            "color_intrinsics": {"width": intr.width, "height": intr.height, "fx": intr.fx, "fy": intr.fy,
                                 "ppx": intr.ppx, "ppy": intr.ppy, "model": str(intr.model),
                                 "coeffs": list(intr.coeffs)},
            "fps": self.fps, "depth_size": list(self.depth_size) if self.depth_on else None,
            "depth_enabled": self.depth_on,
            "usb": dev.get_info(rs.camera_info.usb_type_descriptor)
            if dev.supports(rs.camera_info.usb_type_descriptor) else "?",
        }
        return self

    # -- exposure ----------------------------------------------------------
    def _opt(self, name):
        return getattr(self._rs.option, name)

    def exposure_state(self) -> dict:
        s, st = self._color_sensor, {}
        for key, opt in (("auto_exposure", "enable_auto_exposure"), ("exposure", "exposure"),
                         ("gain", "gain"), ("auto_white_balance", "enable_auto_white_balance"),
                         ("white_balance", "white_balance")):
            o = self._opt(opt)
            if s.supports(o):
                st[key] = s.get_option(o)
        return st

    def toggle_auto(self) -> bool:
        """Switch auto exposure + auto white balance together. Returns the new 'auto' state."""
        st = self.exposure_state()
        new = 0.0 if st.get("auto_exposure", 1.0) else 1.0
        for opt in ("enable_auto_exposure", "enable_auto_white_balance"):
            if self._color_sensor.supports(self._opt(opt)):
                self._color_sensor.set_option(self._opt(opt), new)
        return bool(new)

    def adjust_exposure(self, factor: float) -> float | None:
        """Manual mode only: multiply exposure by factor (clamped to the sensor range)."""
        o = self._opt("exposure")
        if not self._color_sensor.supports(o) or self.exposure_state().get("auto_exposure"):
            return None
        r = self._color_sensor.get_option_range(o)
        val = float(np.clip(self._color_sensor.get_option(o) * factor, r.min, r.max))
        self._color_sensor.set_option(o, val)
        return val

    # -- frames ------------------------------------------------------------
    def grab(self) -> Frame:
        frames = self._pipe.wait_for_frames()
        if self._align is not None:
            frames = self._align.process(frames)
        c = frames.get_color_frame()
        d = frames.get_depth_frame() if self.depth_on else None
        meta = {"timestamp_ms": c.get_timestamp(), "frame_number": c.get_frame_number()}
        md = self._rs.frame_metadata_value
        for key, m in (("actual_exposure", md.actual_exposure), ("gain_level", md.gain_level)):
            if c.supports_frame_metadata(m):
                meta[key] = c.get_frame_metadata(m)
        meta.update(self.exposure_state())
        depth = np.asanyarray(d.get_data()).copy() if d else None
        return Frame(np.asanyarray(c.get_data()).copy(), depth, meta)

    def stop(self) -> None:
        if self._pipe:
            self._pipe.stop()


class UsbCamera:
    """Any extra webcam (OpenCV index), e.g. the gripper camera. Not a RealSense's own RGB index.

    Exposure control over DirectShow is best effort: many cheap cameras ignore it. What the
    driver reports is stored with every frame either way.
    """

    def __init__(self, index: int, size=(640, 480), name: str | None = None, fps: int | None = None,
                 mjpg: bool = True):
        """mjpg=True asks for compressed MJPEG: most webcams give 1080p only in MJPEG (YUYV 1080p is
        often 5 fps) and it uses a fraction of the USB bandwidth."""
        self.index, self.size, self.name, self.fps, self.mjpg = index, size, name or f"cam{index}", fps, mjpg
        self._cap = None
        self.info: dict = {}

    def start(self) -> "UsbCamera":
        self._cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():  # some cameras only open with Media Foundation
            self._cap = cv2.VideoCapture(self.index)
        if self.mjpg:  # must be set before the size
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        if self.fps:
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.index}")
        fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        self.info = {"index": self.index, "width": int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     "height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     "fps": self._cap.get(cv2.CAP_PROP_FPS),
                     "fourcc": "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)) if fourcc else "?"}
        if (self.info["width"], self.info["height"]) != tuple(self.size):
            print(f"[{self.name}] asked {self.size[0]}x{self.size[1]}, camera gives "
                  f"{self.info['width']}x{self.info['height']}")
        return self

    def set_auto(self, on: bool) -> None:
        # DirectShow convention: 0.75 = auto, 0.25 = manual
        self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if on else 0.25)
        self._cap.set(cv2.CAP_PROP_AUTO_WB, 1 if on else 0)

    def exposure_state(self) -> dict:
        return {"auto_exposure_driver": self._cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
                "exposure": self._cap.get(cv2.CAP_PROP_EXPOSURE)}

    def grab(self) -> Frame:
        ok, img = self._cap.read()
        if not ok:
            raise RuntimeError(f"camera {self.index}: no frame")
        return Frame(img, None, self.exposure_state())

    def stop(self) -> None:
        if self._cap:
            self._cap.release()


def depth_preview(depth: np.ndarray | None, size: tuple[int, int], near_mm=100, far_mm=800,
                  scale=0.001) -> np.ndarray:
    """Colour picture of a depth frame for the UI (black = no data). Not for saving."""
    if depth is None:
        img = np.zeros((size[1], size[0], 3), np.uint8)
        cv2.putText(img, "no depth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return img
    mm = depth.astype(np.float32) * scale * 1000
    norm = np.clip((far_mm - mm) / (far_mm - near_mm), 0, 1)  # near = bright
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[depth == 0] = 0
    return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)


def quality(color: np.ndarray) -> dict:
    """Cheap checks shown after every capture: blur (higher = sharper) and clipped highlights."""
    grey = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    return {"sharpness": round(float(cv2.Laplacian(grey, cv2.CV_64F).var()), 1),
            "overexposed_pct": round(100.0 * float((grey >= 250).mean()), 2),
            "brightness": round(float(grey.mean()), 1)}
