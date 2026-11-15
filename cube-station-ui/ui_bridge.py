"""Optional, failure-isolated UI bridge for the SO-101 cube sort station.

Attach this module inside an existing robot runner. This code never opens a camera,
changes ownership, moves an arm directly, or handles low-level servo configuration.
It only forwards explicitly checked key commands to the runner's Keys.feed().

    bridge = UIBridge(key_sink=keys.feed,
                      state_getter=lambda: make_ui_snapshot(sequence),
                      cam_getter=lambda name: hub.cameras[name].read_latest(),
                      root=Path('robot_sequence'))
    bridge.start()  # background daemon; original keyboard remains independent
    bridge.publish('step_started', {...})

The real runner's state_getter MUST provide the current allowed keys in "options".
Missing/invalid options fail closed. HOLD ('q') is always forwarded. The runner
itself remains authoritative over every move and safety interlock.
"""
from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

LOG = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
VALID_COMMANDS = frozenset('qsgbrtxkn')
CAMERAS = {'top', 'gripper', 'side'}


def _options(snapshot: dict) -> list[str]:
    raw = snapshot.get('options', [])
    if not isinstance(raw, list):
        return []
    result = []
    for value in raw:
        key = value if isinstance(value, str) else value.get('key') if isinstance(value, dict) else None
        if isinstance(key, str) and key in VALID_COMMANDS and key != 'q':
            result.append(key)
    return result


def _read_json(filename: Path) -> dict:
    try:
        if filename.exists() and filename.is_file() and filename.stat().st_size < 3_000_000:
            content = json.loads(filename.read_text(encoding='utf-8'))
            return content if isinstance(content, dict) else {'items': content}
    except Exception as e:
        LOG.warning('Cannot read %s: %s', filename, e)
    return {}


def _contained(root: Path, candidate: Path) -> bool:
    return candidate.resolve().is_relative_to(root.resolve())


class Command(BaseModel):
    key: str


class DeriveRequest(BaseModel):
    write: bool = False


class UIBridge:
    def __init__(
        self,
        *,
        key_sink: Callable[[str], Any],
        state_getter: Callable[[], dict[str, Any]],
        cam_getter: Optional[Callable[[str], Any]] = None,
        config_getter: Optional[Callable[[], dict]] = None,
        derive_runner: Optional[Callable[[], Any]] = None,
        root: str | Path = HERE,
        static_dir: str | Path = HERE / 'static',
        run_dir: str | Path | None = None,
        host: str = '127.0.0.1',
        port: int = 8765,
        mode: str = 'loop',
        model_path: str = '',
    ):
        if host != '127.0.0.1':
            raise ValueError('Remote connections are prohibited: use 127.0.0.1 only')
        self.key_sink = key_sink
        self.state_getter = state_getter
        self.cam_getter = cam_getter
        self.config_getter = config_getter
        self.derive_runner = derive_runner
        self.root = Path(root).resolve()
        self.run_dir = (Path(run_dir) if run_dir else self.root / 'runs').resolve()
        self.static_dir = Path(static_dir).resolve()
        self.host = host
        self.port = port
        self.mode = mode
        self.model_path = model_path
        self.bus: deque[tuple[int, dict]] = deque(maxlen=700)
        self.bus_lock = threading.Lock()
        self.sequence_number = 0
        self.server: Optional[uvicorn.Server] = None
        self.thread: Optional[threading.Thread] = None
        self.app = self._make_app()

    def snapshot(self) -> dict:
        try:
            state = self.state_getter()
            if not isinstance(state, dict):
                return {'state': 'UNKNOWN', 'options': [], 'mode': self.mode}
            state = dict(state)
            state['options'] = _options(state)
            state.setdefault('mode', self.mode)
            return state
        except Exception:
            LOG.exception('State callback failed; refusing motion commands')
            return {'state': 'UNKNOWN', 'options': [], 'mode': self.mode}

    def publish(self, event_type: str, payload: Optional[dict] = None, **extra: Any) -> None:
        """May be called by the runner from any thread. Never raises into robot loop."""
        try:
            event = {'type': str(event_type), 'ts': time.time(), **(payload or {}), **extra}
            # Validate serialization before publishing. Ignore untrusted/non-JSON payloads.
            json.dumps(event)
            with self.bus_lock:
                self.sequence_number += 1
                self.bus.append((self.sequence_number, event))
        except Exception:
            LOG.exception('UI event discarded (runner unaffected)')

    def _events_since(self, last: int) -> tuple[int, list[dict]]:
        with self.bus_lock:
            events = [entry for index, entry in self.bus if index > last]
            latest = self.sequence_number
        return latest, events

    def start(self) -> None:
        """Start Uvicorn on a daemon thread. Failure never propagates to robot loop."""
        if self.thread and self.thread.is_alive():
            return
        def serve():
            try:
                config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level='warning', access_log=False)
                self.server = uvicorn.Server(config)
                self.server.run()
            except Exception:
                LOG.exception('UI bridge failed; runner and keyboard remain active')
        try:
            self.thread = threading.Thread(target=serve, daemon=True, name='cube-ui-bridge')
            self.thread.start()
        except Exception:
            LOG.exception('Could not launch optional UI bridge')

    def stop(self) -> None:
        if self.server:
            self.server.should_exit = True

    def _check_origin(self, request: Request) -> None:
        origin = request.headers.get('origin')
        if origin not in (None, f'http://127.0.0.1:{self.port}', f'http://localhost:{self.port}'):
            raise HTTPException(status_code=403, detail='Localhost UI origin required')
        site = request.headers.get('sec-fetch-site')
        if site == 'cross-site':
            raise HTTPException(status_code=403, detail='Cross-site actions blocked')

    def _resolve_run(self, run: str) -> Path:
        if not re.fullmatch(r'[a-zA-Z0-9_.-]+', run) or run in ('.', '..'):
            raise HTTPException(status_code=400, detail='Invalid run name')
        target = self.run_dir / run
        if not _contained(self.run_dir, target):
            raise HTTPException(status_code=403, detail='Outside run archive')
        return target

    def _make_app(self) -> FastAPI:
        api = FastAPI(title='AXIOM local operator UI', docs_url=None, redoc_url=None, openapi_url=None)
        if self.static_dir.exists():
            api.mount('/static', StaticFiles(directory=self.static_dir), name='static')

        @api.get('/')
        def home():
            return FileResponse(self.static_dir / 'index.html', media_type='text/html', headers={'Cache-Control': 'no-store'})

        @api.get('/api/state')
        def state():
            return JSONResponse(self.snapshot(), headers={'Cache-Control': 'no-store'})

        @api.post('/cmd')
        def command(request: Request, body: Command):
            self._check_origin(request)
            key = body.key
            if key not in VALID_COMMANDS or len(key) != 1:
                raise HTTPException(status_code=400, detail='Unknown command')
            current = self.snapshot()
            if key != 'q' and key not in current['options']:
                raise HTTPException(status_code=409, detail='Key not allowed in current runner state')
            try:
                self.key_sink(key)
            except Exception:
                LOG.exception('Keys.feed rejected key %r', key)
                raise HTTPException(status_code=503, detail='Runner rejected command')
            return {'ok': True, 'key': key, 'state': self.snapshot()}

        @api.websocket('/ws')
        async def events(ws: WebSocket):
            # Local network binding plus origin validation protect command-less WebSocket.
            origin = ws.headers.get('origin')
            if origin not in (None, f'http://127.0.0.1:{self.port}', f'http://localhost:{self.port}'):
                await ws.close(code=1008)
                return
            await ws.accept()
            try:
                await ws.send_json({'type': 'hello', 'mode': self.mode, 'model_path': self.model_path,
                                    'steps': ['PICK', 'SNAP_1', 'SIDE2', 'SNAP_2', 'ROLL', 'SNAP_3', 'DECIDE', 'SORT', 'RELEASE', 'HOME'],
                                    'config': self.config_getter() if self.config_getter else {}})
                last, _ = self._events_since(0)
                await ws.send_json({'type': 'state', **self.snapshot()})
                # Polling is cheap and thread-safe; never blocks the state machine.
                while True:
                    await asyncio.sleep(.2)
                    latest, queued = self._events_since(last)
                    for event in queued:
                        await ws.send_json(event)
                    last = latest
            except (WebSocketDisconnect, RuntimeError):
                pass
            except Exception:
                LOG.debug('WebSocket client disconnected', exc_info=True)

        @api.get('/api/done_rules')
        def done_rules():
            return _read_json(self.root / 'done_rules.json')

        @api.get('/api/prompt_stats')
        def prompt_stats():
            return _read_json(self.root / 'prompt_stats.json')

        @api.get('/api/config')
        def config():
            if self.config_getter:
                try:
                    data = self.config_getter()
                    return data if isinstance(data, dict) else {}
                except Exception:
                    LOG.exception('Could not read configuration')
            # Do not import the runner's config.py from a web request.
            return {'binding': '127.0.0.1', 'port': self.port, 'mode': self.mode,
                    'inspection.good_below': 0.825, 'inspection.defective_at_or_above': 0.975,
                    'camera_preview': 'read_latest only', 'commands': 'Keys.feed only'}

        @api.post('/api/derive')
        def derive(request: Request, body: DeriveRequest):
            self._check_origin(request)
            if body.write:
                raise HTTPException(status_code=403, detail='Writing rules is disabled in this UI')
            if not self.derive_runner:
                raise HTTPException(status_code=501, detail='No derive callback configured')
            try:
                return {'report': self.derive_runner()}
            except Exception:
                LOG.exception('Derive callback failed')
                raise HTTPException(status_code=500, detail='Derivation failed')

        @api.get('/api/runs')
        def runs():
            result = []
            if not self.run_dir.is_dir():
                return {'runs': []}
            for directory in sorted(self.run_dir.iterdir(), key=lambda d: d.name, reverse=True):
                if not directory.is_dir() or directory.is_symlink():
                    continue
                cubes = []
                for path in sorted(directory.glob('cube_*/cube.json')):
                    if not _contained(directory, path) or path.is_symlink():
                        continue
                    info = _read_json(path)
                    if info:
                        cubes.append({'cube': info.get('cube'), 'verdict': info.get('verdict'),
                                      'finished': info.get('finished'), 'started': info.get('started')})
                result.append({'name': directory.name, 'cubes': len(cubes), 'items': cubes})
                if len(result) >= 50:
                    break
            return {'runs': result}

        @api.get('/api/runs/{run}/cubes/{n}')
        def cube(run: str, n: int):
            if not 0 <= n <= 999999:
                raise HTTPException(status_code=400, detail='Invalid cube index')
            target = self._resolve_run(run) / f'cube_{n:03d}' / 'cube.json'
            if not _contained(self.run_dir, target) or not target.is_file():
                raise HTTPException(status_code=404, detail='Cube not found')
            report = _read_json(target)
            for snap in report.get('snaps', []):
                if not isinstance(snap, dict):
                    continue
                face, attempt = snap.get('face'), snap.get('attempt', 1)
                if not isinstance(face, int) or not isinstance(attempt, int):
                    continue
                images = {}
                for kind in ('crop', 'heatmap'):
                    image_path = target.parent / f'face{face}_try{attempt}_{kind}.png'
                    if image_path.is_file() and _contained(target.parent, image_path):
                        images[kind] = f'/api/artifacts/{run}/cube_{n:03d}/{image_path.name}'
                if images:
                    snap['images'] = images
            return report

        @api.get('/api/runs/{run}/cubes/{n}/logs')
        def cube_logs(run: str, n: int):
            if not 0 <= n <= 999999:
                raise HTTPException(status_code=400, detail='Invalid cube index')
            folder = self._resolve_run(run) / f'cube_{n:03d}'
            if not _contained(self.run_dir, folder) or not folder.is_dir():
                raise HTTPException(status_code=404, detail='Cube not found')
            result = []
            for path in sorted(folder.glob('log_*.json'))[:30]:
                if not _contained(folder, path) or path.is_symlink():
                    continue
                try:
                    if path.stat().st_size > 4_000_000:
                        continue
                    obj = json.loads(path.read_text(encoding='utf-8'))
                    rows = obj if isinstance(obj, list) else obj.get('rows', obj.get('log', [])) if isinstance(obj, dict) else []
                    if not isinstance(rows, list):
                        continue
                    rows = [r for r in rows if isinstance(r, dict)][:3000]
                    result.append({'name': path.stem, 'rows': rows,
                                   'auto_done_t': obj.get('auto_done_t') if isinstance(obj, dict) else None,
                                   't_operator': obj.get('t_operator') if isinstance(obj, dict) else None})
                except (OSError, ValueError, TypeError):
                    LOG.debug('Skipping unreadable step log: %s', path)
            return {'logs': result}

        @api.get('/api/artifacts/{run}/{filename:path}')
        def artifact(run: str, filename: str):
            # Captures can be viewed, but never written through the browser.
            root = self._resolve_run(run)
            target = root / filename
            if not _contained(root, target) or not target.is_file() or target.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                raise HTTPException(status_code=404, detail='Image not found')
            return FileResponse(target, headers={'Cache-Control': 'no-store'})

        @api.get('/cam/{name}')
        async def camera(name: str):
            if name not in CAMERAS or self.cam_getter is None:
                raise HTTPException(status_code=404, detail='Camera preview not connected')
            def frames():
                try:
                    import cv2
                    import numpy as np
                except ImportError:
                    return
                # Never CameraHub.latest()/fresh(), and never open a camera device.
                while True:
                    try:
                        image = self.cam_getter(name)
                        if isinstance(image, tuple) and len(image) >= 2:
                            image = image[-1]
                        if image is not None:
                            if not isinstance(image, np.ndarray):
                                image = np.asarray(image)
                            if image.ndim == 3 and image.shape[-1] == 3:
                                # read_latest() frames are RGB, JPEG encoder expects BGR.
                                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 68])
                            if ok:
                                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded.tobytes() + b'\r\n'
                    except Exception:
                        LOG.debug('Camera preview skipped frame', exc_info=True)
                    time.sleep(.16)  # <= 6.25 fps, no pressure on owner
            return StreamingResponse(frames(), media_type='multipart/x-mixed-replace; boundary=frame',
                                     headers={'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'})
        return api
