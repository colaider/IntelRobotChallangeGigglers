"""
Jog an SO-101 (SO-ARM101) FOLLOWER arm one servo at a time from the keyboard,
through a Waveshare Bus Servo Adapter.

Six Feetech STS3215 bus servos (IDs 1-6). The Waveshare adapter is a USB <-> TTL
bridge that appears as a COM port on Windows.

Install the library (package name is feetech-servo-sdk, module is scservo_sdk):
    python -m pip install feetech-servo-sdk

Run (port is auto-detected if you leave --port off):
    python so101_jog.py
    python so101_jog.py --port COM7

Controls
--------
    1 2 3 4 5 6   move servo 1..6 FORWARD by one step
    q w e r t y   move servo 1..6 BACKWARD by one step
    + / -         bigger / smaller step size
    p             print all joint positions
    x             enable torque (hold position)
    z             relax / torque off (arm goes limp)
    ESC / Ctrl+C  quit (relaxes the arm first)
"""

import argparse
import sys

import msvcrt  # Windows keyboard input (built in)
from serial.tools import list_ports
from scservo_sdk import PortHandler, PacketHandler, COMM_SUCCESS

# ---- STS3215 control-table addresses --------------------------------------
ADDR_TORQUE_ENABLE = 40
ADDR_ACC = 41
ADDR_GOAL_POSITION = 42      # 2 bytes
ADDR_GOAL_SPEED = 46         # 2 bytes
ADDR_PRESENT_POSITION = 56   # 2 bytes

STS_PROTOCOL_END = 0         # STS series = little-endian

# ---- Config ----------------------------------------------------------------
BAUDRATE = 1000000
SERVO_IDS = [1, 2, 3, 4, 5, 6]
POS_MIN, POS_MAX = 0, 4095
DEFAULT_STEP = 50            # ticks per key press (~4.4 deg)
MOVE_SPEED = 500             # ticks/s
MOVE_ACC = 30

FORWARD_KEYS = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6}
BACKWARD_KEYS = {"q": 1, "w": 2, "e": 3, "r": 4, "t": 5, "y": 6}

ERROR_BITS = {1: "voltage", 2: "angle limit", 4: "overheat",
              8: "overcurrent", 32: "overload"}


def describe_error(err):
    return ", ".join(name for bit, name in ERROR_BITS.items() if err & bit)


def find_port():
    """Return the adapter's COM port, or exit with a helpful message."""
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found - Windows can't see the Waveshare adapter.\n"
              "  * Plug the adapter into USB (try another cable: some are charge-only).\n"
              "  * Check Device Manager > Ports (COM & LPT). If it shows up under\n"
              "    'Other devices' with a warning icon, install the CH343 USB-serial\n"
              "    driver from Waveshare's Bus Servo Adapter wiki page.")
        sys.exit(1)
    if len(ports) == 1:
        print("Using {} ({})".format(ports[0].device, ports[0].description))
        return ports[0].device
    print("Several ports found - rerun with --port <name>:")
    for p in ports:
        print("  {:8} {}".format(p.device, p.description))
    sys.exit(1)


class Arm:
    def __init__(self, port, baud):
        self.ph = PortHandler(port)
        self.pk = PacketHandler(STS_PROTOCOL_END)
        self.port = port
        self.baud = baud
        self.alive = []
        self.goal = {}

    def connect(self):
        if not self.ph.openPort():
            sys.exit("Could not open {}. Is another program (e.g. a servo tool) "
                     "using it?".format(self.port))
        if not self.ph.setBaudRate(self.baud):
            sys.exit("Could not set baudrate {}".format(self.baud))
        print("Opened {} @ {} baud".format(self.port, self.baud))

    def close(self):
        self.ph.closePort()

    def present(self, sid):
        """Present position in ticks, or None if the servo didn't answer.
        Status error bits (e.g. low voltage) are reported, not treated as dead."""
        pos, comm, err = self.pk.read2ByteTxRx(self.ph, sid, ADDR_PRESENT_POSITION)
        if comm != COMM_SUCCESS:
            return None
        if err:
            print("  ! servo {} reports: {}".format(sid, describe_error(err)))
        return pos

    def scan(self):
        for sid in SERVO_IDS:
            pos = self.present(sid)
            if pos is None:
                print("  ID {}: no response".format(sid))
            else:
                print("  ID {}: OK at {} ticks".format(sid, pos))
                self.alive.append(sid)
                self.goal[sid] = pos  # start from where it is: no jump

    def setup_motion_profile(self):
        for sid in self.alive:
            self.pk.write1ByteTxRx(self.ph, sid, ADDR_ACC, MOVE_ACC)
            self.pk.write2ByteTxRx(self.ph, sid, ADDR_GOAL_SPEED, MOVE_SPEED)

    def hold_here(self):
        """Set every goal to the current position, then enable torque.
        Prevents the arm snapping to a stale target left in the servo."""
        for sid in self.alive:
            pos = self.present(sid)
            if pos is not None:
                self.goal[sid] = pos
                self.pk.write2ByteTxRx(self.ph, sid, ADDR_GOAL_POSITION, pos)
        self.torque(True)

    def torque(self, on):
        for sid in self.alive:
            self.pk.write1ByteTxRx(self.ph, sid, ADDR_TORQUE_ENABLE, 1 if on else 0)
        print("Torque {}".format("ON" if on else "OFF (arm is limp)"))

    def move(self, sid, target):
        target = max(POS_MIN, min(POS_MAX, int(target)))
        comm, err = self.pk.write2ByteTxRx(self.ph, sid, ADDR_GOAL_POSITION, target)
        if comm != COMM_SUCCESS:
            print("servo {}: write failed ({})".format(sid, self.pk.getTxRxResult(comm)))
            return
        self.goal[sid] = target
        edge = "  (at limit)" if target in (POS_MIN, POS_MAX) else ""
        print("servo {} -> {} ticks{}".format(sid, target, edge))

    def print_positions(self):
        for sid in SERVO_IDS:
            pos = self.present(sid)
            if pos is None:
                print("  ID {}: no response".format(sid))
            else:
                print("  ID {}: {:>4} ticks ({:6.1f} deg)".format(
                    sid, pos, pos * 360.0 / 4096))


def read_key():
    """One keypress as a lowercase string. Extended keys (arrows, F-keys) -> ''."""
    ch = msvcrt.getch()
    if ch in (b"\x00", b"\xe0"):
        msvcrt.getch()  # swallow the second byte
        return ""
    return ch.decode("latin-1").lower()


def main():
    ap = argparse.ArgumentParser(description="Jog SO-101 follower servos")
    ap.add_argument("--port", help="COM port (auto-detected if omitted)")
    ap.add_argument("--baud", type=int, default=BAUDRATE)
    args = ap.parse_args()

    arm = Arm(args.port or find_port(), args.baud)
    arm.connect()

    print("Scanning servos...")
    arm.scan()
    if not arm.alive:
        print("No servos answered. Check the 12V supply to the arm, the 3-pin bus "
              "cable, and that the baudrate is {}.".format(args.baud))
        arm.close()
        sys.exit(1)

    arm.setup_motion_profile()
    arm.hold_here()
    step = DEFAULT_STEP
    print(__doc__[__doc__.index("Controls"):])
    print("Step = {} ticks. Ready.\n".format(step))

    try:
        while True:
            key = read_key()
            if key in ("\x1b", "\x03"):  # ESC or Ctrl+C
                break
            elif key in FORWARD_KEYS or key in BACKWARD_KEYS:
                sid = FORWARD_KEYS.get(key) or BACKWARD_KEYS[key]
                if sid not in arm.alive:
                    print("servo {} is not responding - skipped".format(sid))
                    continue
                delta = step if key in FORWARD_KEYS else -step
                arm.move(sid, arm.goal[sid] + delta)
            elif key in ("+", "="):
                step = min(500, step + 10)
                print("step = {}".format(step))
            elif key in ("-", "_"):
                step = max(5, step - 10)
                print("step = {}".format(step))
            elif key == "p":
                arm.print_positions()
            elif key == "x":
                arm.hold_here()
            elif key == "z":
                arm.torque(False)
    finally:
        arm.torque(False)
        arm.close()
        print("Closed. Arm relaxed.")


if __name__ == "__main__":
    main()
