"""SO-101 follower pose reader: torque off (move the arm by hand), read all joint positions.

Feetech STS3215 bus servos (IDs 1-6) through the Waveshare Bus Servo Adapter (a COM port).
Positions are 0..4095 ticks per turn; 2048 = middle. Degrees here are simply
(ticks - 2048) * 360 / 4096, NOT the calibrated angles Physical AI Studio / LeRobot use, so
the raw ticks are always saved too (calibration can be applied later).

    with So101Reader() as arm:      # auto-detects the COM port, torque OFF -> arm goes limp
        pose = arm.read()           # {"shoulder_pan": {"ticks": 2051, "deg": 0.3}, ...}

Only one program can open the COM port: close Physical AI Studio / so101_jog.py first.
"""

import time

from serial.tools import list_ports

ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42  # 2 bytes
ADDR_PRESENT_POSITION = 56  # 2 bytes
BAUDRATE = 1_000_000
JOINTS = {  # servo id -> name (SO-101 / LeRobot naming)
    1: "shoulder_pan",
    2: "shoulder_lift",
    3: "elbow_flex",
    4: "wrist_flex",
    5: "wrist_roll",
    6: "gripper",
}


def find_port() -> str:
    """The adapter's COM port if exactly one serial port exists, else raise with a hint."""
    ports = list(list_ports.comports())
    if len(ports) == 1:
        return ports[0].device
    if not ports:
        raise RuntimeError("No serial port: plug in the Waveshare adapter (data cable, CH343 driver).")
    names = ", ".join(f"{p.device} ({p.description})" for p in ports)
    raise RuntimeError(f"Several serial ports, pass port=...: {names}")


def ticks_to_deg(ticks: int) -> float:
    return (ticks - 2048) * 360.0 / 4096.0


class So101Reader:
    """Read joint positions of the SO-101 follower; torque is switched off so you can pose it.

    torque_off=True (default) makes the arm limp: HOLD IT when connecting, it will drop.
    The arm is left limp on close (safe: it never jumps to an old goal position).
    """

    def __init__(self, port: str | None = None, baud: int = BAUDRATE, torque_off: bool = True):
        from scservo_sdk import PacketHandler, PortHandler  # pip install feetech-servo-sdk

        self.port = port or find_port()
        self.baud = baud
        self.torque_off = torque_off
        self._ph = PortHandler(self.port)
        self._pk = PacketHandler(0)  # 0 = little-endian STS protocol
        self.alive: list[int] = []
        self.holding = False

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> "So101Reader":
        if not self._ph.openPort():
            raise RuntimeError(f"Cannot open {self.port} (in use by another program?)")
        if not self._ph.setBaudRate(self.baud):
            raise RuntimeError(f"Cannot set baud rate {self.baud}")
        self.alive = [sid for sid in JOINTS if self._read_ticks(sid) is not None]
        missing = [JOINTS[s] for s in JOINTS if s not in self.alive]
        if not self.alive:
            self._ph.closePort()
            raise RuntimeError("No servo answers: check the arm's 12 V power and the bus cable.")
        if missing:
            print(f"[robot] WARNING: no answer from {missing}")
        if self.torque_off:
            self.set_torque(False)
        return self

    def close(self) -> None:
        try:
            self.set_torque(False)
        finally:
            self._ph.closePort()

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    # -- servo access ------------------------------------------------------
    def _read_ticks(self, sid: int) -> int | None:
        from scservo_sdk import COMM_SUCCESS

        value, comm, _err = self._pk.read2ByteTxRx(self._ph, sid, ADDR_PRESENT_POSITION)
        return value if comm == COMM_SUCCESS else None

    def set_torque(self, on: bool) -> None:
        for sid in self.alive:
            self._pk.write1ByteTxRx(self._ph, sid, ADDR_TORQUE_ENABLE, 1 if on else 0)
        self.holding = on

    def hold(self, on: bool) -> None:
        """on: freeze the arm where it is (goal := present position, then torque on), so you can
        take your hands out of the picture. off: limp again. Goal is written first so the arm
        never jumps to an old target."""
        if on:
            for sid in self.alive:
                ticks = self._read_ticks(sid)
                if ticks is not None:
                    self._pk.write2ByteTxRx(self._ph, sid, ADDR_GOAL_POSITION, ticks)
        self.set_torque(on)

    def read(self) -> dict:
        """All joints: {name: {"id", "ticks", "deg"}} plus "t" (time.time()). Missing -> ticks None."""
        pose = {"t": time.time()}
        for sid, name in JOINTS.items():
            ticks = self._read_ticks(sid) if sid in self.alive else None
            pose[name] = {"id": sid, "ticks": ticks, "deg": None if ticks is None else round(ticks_to_deg(ticks), 2)}
        return pose


def short(pose: dict | None) -> str:
    """One-line summary for the UI: pan 12.3 lift -40.1 ..."""
    if not pose:
        return "robot: off"
    abbr = {"shoulder_pan": "pan", "shoulder_lift": "lift", "elbow_flex": "elb",
            "wrist_flex": "wfl", "wrist_roll": "wrl", "gripper": "grip"}
    return "  ".join(f"{abbr[n]} {pose[n]['deg']:.0f}" if pose[n]["deg"] is not None else f"{abbr[n]} --"
                     for n in JOINTS.values())


if __name__ == "__main__":  # quick check: py collection/robot.py --port COM6  -> prints the pose 10x
    import argparse

    ap = argparse.ArgumentParser(description="Torque off and print the arm pose 10 times (HOLD THE ARM).")
    ap.add_argument("--port", default=None)
    with So101Reader(port=ap.parse_args().port) as arm:
        print(f"Connected on {arm.port}, torque OFF - move the arm by hand")
        for _ in range(10):
            print(short(arm.read()))
            time.sleep(0.5)
