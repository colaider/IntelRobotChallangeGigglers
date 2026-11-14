"""Dataset collector: live view of all cameras + depth + robot pose, capture on a key press.

Run from the repo root with the system Python (needs a GUI OpenCV, not the Anomalib .venv):
    py collection/run.py --list                                   # which RealSense serials / USB indexes exist
    py collection/run.py --rs ground=123456789012 top=987654321098 --usb gripper=2 --name red_good_a
    py collection/run.py                                          # every RealSense (rs1, rs2...), no USB cams
    py collection/run.py --no-robot --label defective --defects dot line scratch

Collect at the resolution you will RUN at (default 640x480 everywhere): on a RealSense, 640x480
is 4:3 and sees a narrower field than 1280x720 (16:9), so other modes give other pictures.

Keys (click the window first):
    SPACE      capture (all cameras + depth + robot pose)
    h          hold the arm where it is / let it go limp again (pose by hand, then h, hands out)
    g / d      label good / defective        1-9  defect type (from --defects)
    a          auto exposure + white balance ON/OFF on all cameras (collect with OFF)
    [ / ]      exposure down / up (RealSense, manual mode)
    BACKSPACE  delete the last capture       q / ESC  quit (arm goes limp: hold it!)
"""

import argparse
import os
import sys
import time
from pathlib import Path

if "--list" in sys.argv:  # probing camera indexes prints pages of harmless OpenCV warnings
    os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
    os.environ["OPENCV_VIDEOIO_PRIORITY_OBSENSOR"] = "0"

import cv2  # noqa: E402
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cameras import RealSenseCamera, UsbCamera, depth_preview, list_realsense, list_usb, quality  # noqa: E402
from robot import So101Reader, short  # noqa: E402
from session import SessionWriter  # noqa: E402

TILE_W = 480  # width of each camera tile in the window


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="print connected RealSense serials and USB indexes, exit")
    p.add_argument("--out", type=Path, default=Path("data"), help="sessions root (gitignored)")
    p.add_argument("--name", default="", help="session name, e.g. red_good_a")
    p.add_argument("--notes", default="", help="free text stored in session.json (light, setup...)")
    p.add_argument("--label", choices=["good", "defective"], default="good")
    p.add_argument("--color", default="", help="brick colour for this session (optional)")
    p.add_argument("--defects", nargs="*", default=["dot", "line", "scratch", "crack", "chip", "other"],
                   help="defect names for keys 1-9")
    p.add_argument("--rs", nargs="*", default=None, metavar="NAME=SERIAL",
                   help="RealSense cameras to use, e.g. ground=1234 top=5678 (default: all, named rs1, rs2...)")
    p.add_argument("--no-realsense", action="store_true")
    p.add_argument("--usb", nargs="*", default=[], metavar="NAME=INDEX",
                   help="extra USB cameras, e.g. gripper=2 (or just 2)")
    p.add_argument("--size", default="640x480", help="colour resolution WxH for every camera")
    p.add_argument("--rs-depth", default="640x480", help="RealSense depth resolution WxH")
    p.add_argument("--no-robot", action="store_true")
    p.add_argument("--port", default=None, help="robot COM port (auto if only one)")
    p.add_argument("--manual-exposure", action="store_true", help="start with auto exposure/WB OFF")
    p.add_argument("--min-sharpness", type=float, default=30,
                   help="warn below this Laplacian variance (scene dependent: plain backgrounds score low)")
    return p.parse_args(argv)


def wh(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def named(items: list[str], prefix: str) -> list[tuple[str, str]]:
    """['top=123', '456'] -> [('top', '123'), ('<prefix>456', '456')]"""
    out = []
    for it in items:
        name, _, val = it.partition("=")
        out.append((name, val) if val else (f"{prefix}{name}", name))
    return out


def print_devices() -> None:
    try:
        rs = list_realsense()
    except ImportError:
        rs = []
        print("pyrealsense2 not installed")
    print("RealSense:" if rs else "RealSense: none found")
    for d in rs:
        print(f"  serial {d['serial']}  {d['model']}  USB {d['usb']}" + ("   <- USB 2: plug into a blue USB 3 port"
                                                                      if str(d["usb"]).startswith("2") else ""))
    print(f"USB camera indexes that open: {list_usb()}  (a RealSense's colour image shows up here too - "
          "try each with --usb test=N --no-realsense --no-robot to see which is the gripper)")
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    print("Serial ports:" if ports else "Serial ports: none (servo adapter not connected?)")
    for p in ports:
        print(f"  {p.device:7s} {p.description}")
    if len(ports) > 1:
        print("  several ports: pass --port COMx for the FOLLOWER arm (py collection/robot.py --port COMx "
              "prints its pose; the arm you move should change)")


def open_devices(args):
    """Returns (cameras, robot). Each part that fails is reported and skipped."""
    size, cams = wh(args.size), []
    if not args.no_realsense:
        try:
            pairs = (named(args.rs, "rs") if args.rs
                     else [(f"rs{i + 1}", d["serial"]) for i, d in enumerate(list_realsense())])
        except Exception as e:  # noqa: BLE001
            pairs = []
            print(f"[realsense] not used: {e}")
        for name, serial in pairs:
            try:
                cams.append(RealSenseCamera(name, serial, color=size, depth=wh(args.rs_depth)).start())
                print(f"[{name}] RealSense {serial} ok")
            except Exception as e:  # noqa: BLE001 - show any device problem, keep going
                print(f"[{name}] RealSense {serial} not used: {e}")
    for name, idx in named(args.usb, "cam"):
        try:
            cams.append(UsbCamera(int(idx), size, name).start())
            print(f"[{name}] USB camera {idx} ok")
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] USB camera {idx} not used: {e}")
    robot = None
    if not args.no_robot:
        try:
            robot = So101Reader(port=args.port).connect()
            print(f"[robot] {robot.port}: torque OFF, move the arm by hand")
        except Exception as e:  # noqa: BLE001
            print(f"[robot] not used: {e}")
    return cams, robot


def tile(img: np.ndarray, title: str) -> np.ndarray:
    th = int(TILE_W * img.shape[0] / img.shape[1])
    t = cv2.resize(img, (TILE_W, th))
    cv2.putText(t, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(t, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return t


def compose(frames, cams, status: list[str], flash: bool) -> np.ndarray:
    """Row 1: every camera's colour image. Row 2: depth of each RealSense. Status text below."""
    row1 = [tile(frames[c.name].color, c.name) for c in cams]
    h1 = max(t.shape[0] for t in row1)
    row1 = [np.vstack([t, np.zeros((h1 - t.shape[0], TILE_W, 3), np.uint8)]) for t in row1]
    rows = [np.hstack(row1)]
    if any(frames[c.name].depth is not None for c in cams):
        row2 = []
        for c in cams:
            d = frames[c.name].depth
            scale = c.info.get("depth_scale", 0.001) if isinstance(c.info, dict) else 0.001
            img = depth_preview(d, (TILE_W, h1), scale=scale) if d is not None else np.zeros((h1, TILE_W, 3), np.uint8)
            row2.append(tile(img, f"{c.name} depth") if d is not None else img)
        rows.append(np.hstack(row2))
    canvas = np.vstack(rows)
    bar = np.zeros((26 * len(status) + 10, canvas.shape[1], 3), np.uint8)
    for i, line in enumerate(status):
        cv2.putText(bar, line, (10, 24 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    out = np.vstack([canvas, bar])
    if flash:
        cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), (255, 255, 255), 12)
    return out


def set_auto(cams, on: bool) -> None:
    for c in cams:
        if isinstance(c, RealSenseCamera):
            if bool(c.exposure_state().get("auto_exposure", 1.0)) != on:
                c.toggle_auto()
        else:
            c.set_auto(on)


def exposure_text(rs_cams) -> str:
    if not rs_cams:
        return "exposure: no RealSense"
    parts = []
    for c in rs_cams:
        st = c.exposure_state()
        parts.append(f"{c.name} " + ("AUTO" if st.get("auto_exposure") else f"{st.get('exposure', 0):.0f}"))
    return "exposure: " + "  ".join(parts)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.list:
        print_devices()
        return
    cams, robot = open_devices(args)
    if not cams:
        raise SystemExit("No camera available - nothing to collect (try --list).")
    rs_cams = [c for c in cams if isinstance(c, RealSenseCamera)]
    auto = not args.manual_exposure
    if not auto:
        set_auto(cams, False)

    writer = SessionWriter(args.out, args.name, args.notes, cameras={c.name: c.info for c in cams},
                           robot={"port": robot.port, "alive_ids": robot.alive} if robot else {})
    print(f"Session: {writer.dir}")
    label, defect, msg, flash_until = args.label, "", "SPACE = capture", 0.0
    try:
        while True:
            frames = {c.name: c.grab() for c in cams}
            pose = robot.read() if robot else None
            status = [
                f"{writer.dir.name}   captures: {writer.count}   label: {label.upper()}"
                + (f" ({defect})" if label == "defective" else "") + (f"   colour: {args.color}" if args.color else ""),
                f"{exposure_text(rs_cams)}   |   {short(pose)}" + ("   [HOLDING]" if robot and robot.holding else ""),
                msg,
            ]
            cv2.imshow("collector", compose(frames, cams, status, time.time() < flash_until))
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                if label == "defective" and not defect:
                    msg = "Pick a defect type first (keys 1-9): " + " ".join(
                        f"{i + 1}={d}" for i, d in enumerate(args.defects[:9]))
                    continue
                row = writer.save(frames, pose, label, defect if label == "defective" else "", args.color)
                warn = []
                for c in cams:
                    q = quality(frames[c.name].color)
                    if q["overexposed_pct"] > 1.0:
                        warn.append(f"{c.name} OVEREXPOSED {q['overexposed_pct']}%")
                    if q["sharpness"] < args.min_sharpness:
                        warn.append(f"{c.name} low sharpness {q['sharpness']:.0f}")
                if auto:
                    warn.append("auto exposure is ON")
                msg = f"saved {row['id']}  " + ("  ".join(warn) if warn else "ok")
                flash_until = time.time() + 0.15
            elif key == ord("g"):
                label, msg = "good", "label: good"
            elif key == ord("d"):
                label, msg = "defective", "label: defective - pick defect type 1-9: " + " ".join(
                    f"{i + 1}={d}" for i, d in enumerate(args.defects[:9]))
            elif ord("1") <= key <= ord("9") and key - ord("1") < len(args.defects):
                defect = args.defects[key - ord("1")]
                msg = f"defect type: {defect}"
            elif key == ord("h") and robot:
                robot.hold(not robot.holding)
                msg = "arm HOLDING - hands out of the picture" if robot.holding else "arm limp - hold it"
            elif key == ord("a"):
                auto = not auto
                set_auto(cams, auto)
                msg = "auto exposure + WB " + ("ON" if auto else "OFF (fixed) on all cameras")
            elif key in (ord("["), ord("]")) and rs_cams:
                vals = [c.adjust_exposure(0.8 if key == ord("[") else 1.25) for c in rs_cams]
                msg = ("turn auto exposure off first (a)" if all(v is None for v in vals)
                       else "exposure " + "  ".join(f"{c.name} {v:.0f}" for c, v in zip(rs_cams, vals) if v))
            elif key == 8:  # backspace
                cid = writer.delete_last()
                msg = f"deleted {cid}" if cid else "nothing to delete"
            elif key in (ord("q"), 27):
                break
    finally:
        cv2.destroyAllWindows()
        for c in cams:
            c.stop()
        if robot:
            robot.close()
        print(f"{writer.count} captures in {writer.dir}")


if __name__ == "__main__":
    main()
