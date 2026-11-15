# AXIOM — SO-101 cube sort operator UI

**Run the fully functional, hardware-free simulator**

Double-click **start.sh** (Linux/macOS, using Terminal) or **start.bat** (Windows); the browser opens automatically. Or run `python launch.py`. The local address is **http://127.0.0.1:8765**. Requirements: `fastapi`, `uvicorn` (the physical-ai-studio backend venv already includes these). For camera previews when connected to a real runner, `numpy` and `opencv-python` are needed.

Use **Start next cube** → confirm to run the synthetic pipeline. Use **HOLD** (or `Q`) at any time. The simulator generates synthetic camera frames, live rules diagnostics, inspections, counts, and saved demo run reports. Demo data lives in `demo_runs/` and is not hardware telemetry.

**Connecting the real robot**

The actual `robot_sequence/` sources were not provided, so the runner integration below is an adapter pattern, not a verified patch. Copy `static/` and `ui_bridge.py` into your runner project. Attach one optional bridge in the same Python process after creating the existing `Sequence`, `Keys` and `CameraHub` objects:

```python
from ui_bridge import UIBridge

bridge = UIBridge(
    key_sink=keys.feed,
    state_getter=lambda: {
        'state': sequence.state.name,
        'options': allowed_keys_for_current_state(),  # from runner's authoritative state
        'cube_n': sequence.cube_n,
        'counts': counts_from_report(),                # optional
        'faces': snapshots_for_current_cube(),        # optional
    },
    cam_getter=lambda name: camera_hub.cameras[{
        'top': 'camera top', 'gripper': 'camera gripper', 'side': 'camera side'
    }[name]].read_latest(),
    root='robot_sequence',
    mode='loop',
)
bridge.start()
```

Instrument existing runner transitions using `bridge.publish('state', {...})`, `bridge.publish('step_started', {...})`, `bridge.publish('step_status', {...})`, `bridge.publish('snap', {...})`, `bridge.publish('decision', {...})`, `bridge.publish('failure', {...})`, and `bridge.publish('run_done', {...})` with the payload shapes described in UI_CONTEXT.md. Keep the terminal `Sequence.say()` as ground truth; in addition call `bridge.publish('log', {'message': msg})`. Never parse `say()` lines for the UI.

The bridge is optional, daemon-threaded, and catches event/preview exceptions. If it fails, the robot loop and keyboard continue. All commands go through `Keys.feed`; the server refuses keys not in the current `options` except HOLD (`q`). Only HTTP POST `/cmd` from localhost UI can trigger actions; page mount, reload, WebSocket reconnect, navigation and camera fetches are entirely read-only. The UI confirms all standby movement (`g`, `r`, `b`, `t`). The physical power switch remains the emergency stop.

## Available screens

- **Live station**: sequence, three passive camera previews, command gating, HOLD, done-rule telemetry, inspection, anomaly bands (0.825/0.975), temperature warnings (60 °C), counts and event stream.
- **Teach mode**: operator `n` only during manual VLA; post-confirmation recording indicator, read-only rule analysis callback.
- **Snap test**: large crop and heatmap, verdict, thresholds, latency and debug-step command only when permitted.
- **Run review**: browse run/cube records, captured face images, step durations, and per-tick trend/wobble/pose error traces with auto-DONE/operator markers.
- **Configuration**: current settings and done rules, intentionally read-only.

## Safety and intentional limitations

- No automatic movement and no UI-triggered robot process startup.
- No remote bind; the bridge enforces `127.0.0.1`.
- Never writes servo parameters, EEPROM, camera ownership, calibration, training or `done_rules.json`.
- HOLD is a software-controlled hold, **not** a replacement for the real emergency switch.
- The browser cannot infer `options` from text logs. Real integration MUST supply current allowed keys.
- The browser's preview is synthetic in the standalone demo. Real camera endpoints use only non-destructive `read_latest()`.
- The given Claude artifact URL could not be fetched in this environment, so the delivered layout follows UI_CONTEXT.md rather than copying that artifact.
