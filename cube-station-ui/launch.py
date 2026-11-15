"""Launch the fully interactive, synthetic localhost demo with zero robot dependencies.

    python launch.py
    # open http://127.0.0.1:8765

No robot process, servo commands, camera access or hardware movement occurs.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import time
from ui_bridge import UIBridge
from demo_engine import SimStation


def main():
    parser = argparse.ArgumentParser(description='AXIOM cube station local simulator')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--mode', choices=('sim','debug'), default='sim')
    parser.add_argument('--no-browser', action='store_true', help='Do not automatically open browser')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    proxy = {'station': None}

    def key(k):
        if proxy['station']:
            proxy['station'].key(k)

    def snapshot():
        return proxy['station'].snapshot() if proxy['station'] else {'state': 'STANDBY','options': []}

    bridge = UIBridge(key_sink=key, state_getter=snapshot, root=here,
                      run_dir=here / 'demo_runs', host='127.0.0.1', port=args.port,
                      mode='sim' if args.mode=='sim' else 'debug', model_path='SIMULATED / SmolVLA',
                      config_getter=lambda: {'station': 'SO-101 / simulated', 'runner': args.mode,
                                             'inspection.model': 'PatchCore front_all (simulated)',
                                             'inspection.good_below': 0.825,
                                             'inspection.defective_at_or_above': 0.975,
                                             'policy': 'SmolVLA (simulated)',
                                             'camera_preview_fps': 6,
                                             'backend': '127.0.0.1',
                                             'robot_hardware': 'NOT CONNECTED'},
                      derive_runner=lambda: {'mode': 'SIMULATED', 'agree': 8, 'early': 1,
                                             'late': 1, 'never': 0, 'stuck': 0,
                                             'write_performed': False})
    proxy['station'] = SimStation(bridge.publish, here/'demo_runs',mode=args.mode)
    print(f'AXIOM station SIMULATOR running at http://127.0.0.1:{args.port}')
    print('All feed imagery and measurements are synthetic. Physical robot NOT connected.')
    print('Press Ctrl+C to exit. UI HOLD simulates the runner keyboard Q action.')
    # Browser opens only for the standalone simulator, never during real runner attach.
    if not args.no_browser:
        import threading, webbrowser
        def open_when_ready():
            time.sleep(1.8)
            try:
                webbrowser.open(f'http://127.0.0.1:{args.port}')
            except Exception:
                pass
        threading.Thread(target=open_when_ready, daemon=True).start()
    # Uvicorn runs on foreground here; demo process has nothing else to preserve.
    import uvicorn
    uvicorn.run(bridge.app, host='127.0.0.1', port=args.port, log_level='warning', access_log=False)

if __name__ == '__main__':
    main()
