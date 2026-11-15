"""Safe, synthetic robot sequence. No robot hardware or cameras required."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import json
import math
import random
import threading
import time

STEP_DEFS = [
    ('PICK', 'VLA', 3.3, 'Pick up the cube and show front side to the Side Camera'),
    ('SNAP_1', 'SNAP', 1.15, ''),
    ('SIDE2', 'VLA', 3.0, 'Show side 2 to the Side Camera'),
    ('SNAP_2', 'SNAP', 1.15, ''),
    ('ROLL', 'SCRIPT', 1.65, 'Rotate wrist +180°'),
    ('SNAP_3', 'SNAP', 1.15, ''),
    ('DECIDE', 'DECIDE', 0.65, ''),
    ('SORT', 'VLA', 2.8, ''),
    ('RELEASE', 'SCRIPT', 0.75, 'Open gripper'),
    ('HOME', 'SCRIPT', 1.8, 'Return to start pose'),
]


class SimStation:
    """Sim loop never begins movement until explicit command `g` (or r/t/b)."""
    def __init__(self, publisher, run_dir: Path, mode='sim'):
        self.publish = publisher
        self.run_dir = run_dir
        self.mode = mode
        self.lock = threading.RLock()
        self.state = 'STANDBY'
        self.options = ['g', 'r', 'x']
        self.cube_n = 1
        self.counts = {'total': 0, 'good': 0, 'defective': 0}
        self.cycle_times = []
        self.cycle_started = 0
        self.run_name = 'demo_' + datetime.now().strftime('%Y%m%dT%H%M%S')
        self.index = -1
        self.step = None
        self.kind = None
        self.prompt = ''
        self.timeout = 0
        self.started = 0
        self.temps = [37, 41, 39, 36, 35, 34]
        self.faces = []
        self.latest_snap = None
        self.steps = []
        self.manual = False
        self.manual_pending_until = 0
        self.stop_after = False
        self.failed_step = None
        self.cancelled = False
        self.snap_retries = {}
        self.t = 0
        self.pose_err = 0
        self.trend = 0
        self.wobble = 0
        self.still = 0
        self.flat = 0
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True, name='sim-station')
        self.thread.start()
        self.publish('log', {'message': 'Simulation armed. No movement until operator starts a cube.'})

    def snapshot(self):
        with self.lock:
            return {'mode': 'sim' if self.mode == 'sim' else 'debug', 'state': self.state,
                    'options': self.options.copy(), 'cube_n': self.cube_n,
                    'counts': self.counts.copy(),
                    'cycle_time': round(sum(self.cycle_times)/len(self.cycle_times), 1) if self.cycle_times else None,
                    'step': self.step, 'kind': self.kind, 'prompt': self.prompt,
                    'timeout_s': self.timeout, 't': round(self.t, 2), 'pose_err': round(self.pose_err, 3),
                    'trend_mm_s': round(self.trend, 2), 'wobble_mm': round(self.wobble, 2),
                    'manual': self.manual, 'faces': [dict(s) for s in self.faces],
                    'latest_snap': dict(self.latest_snap) if self.latest_snap else None,
                    'temps': self.temps.copy(), 'run_dir': str(self.run_dir / self.run_name)}

    def _emit_state(self):
        self.publish('state', {'state': self.state, 'options': self.options.copy(),
                               'cube_n': self.cube_n, 'counts': self.counts.copy(),
                               'faces': [dict(s) for s in self.faces],
                               'cycle_time': round(sum(self.cycle_times)/len(self.cycle_times), 1) if self.cycle_times else None})

    def _enter(self, index):
        self.index = index
        self.step, self.kind, duration, prompt = STEP_DEFS[index]
        if self.step == 'SORT':
            verdict = self._verdict()
            prompt = 'move cube to the right bowl' if verdict == 'good' else 'move cube to the left bowl'
        self.prompt = prompt
        self.timeout = duration + 9
        self.started = time.monotonic()
        self.t = 0
        self.state = self.step
        self.manual = self.mode == 'debug' and self.kind == 'VLA'
        self.manual_pending_until = 0
        self.options = ['q', 's'] + (['n'] if self.manual else [])
        self.pose_err = 0.5
        self.trend = 5
        self.wobble = 2
        self.still = 0
        self.flat = 0
        self.publish('step_started', {'step': self.step, 'kind': self.kind, 'prompt': prompt,
                                      'attempt': 1, 'timeout_s': self.timeout, 'rules_source': 'simulated',
                                      'manual': self.manual})
        self._emit_state()

    def _start_cube(self):
        self.faces = []
        self.snap_retries = {}
        self.latest_snap = None
        self.steps = []
        self.cycle_started = time.monotonic()
        self.cancelled = False
        self.failed_step = None
        self._enter(0)
        self.publish('log', {'message': f'Cube {self.cube_n} started (SIMULATED).'})

    def key(self, key):
        with self.lock:
            if key == 'q':
                if self.state != 'STANDBY':
                    failed = self.step
                    self.failed_step = self.index
                    self.cancelled = True
                    self.state = 'STANDBY'
                    self.step = None
                    self.kind = None
                    self.prompt = ''
                    self.options = ['b', 't', 'r', 'x']
                    self.publish('failure', {'step': failed, 'code': 'HOLD',
                                             'message': 'Operator HOLD: arm remains in current pose (simulated).',
                                             'options': self.options.copy()})
                else:
                    self.publish('log', {'message': 'HOLD received in STANDBY. No motion.'})
                self._emit_state()
                return
            if key not in self.options:
                return  # Bridge rejects first, runner still authoritative.
            if key == 'g' and self.state == 'STANDBY':
                if self.mode == 'sim':
                    self._start_cube()
                else:
                    self._start_cube()
            elif key == 'r' and self.state == 'STANDBY':
                self.faces = []
                self.cancelled = True
                self._enter(9)
                self.publish('log', {'message': 'Operator commanded slow return to start pose (simulated).'})
            elif key == 'b' and self.state == 'STANDBY':
                self._enter(9)
                self.publish('log', {'message': 'Back command simulated by safe return to start pose.'})
            elif key == 't' and self.state == 'STANDBY':
                if self.failed_step is not None:
                    self._enter(self.failed_step)
            elif key == 's':
                self.stop_after = True
                self.publish('log', {'message': 'Stop after current cube requested.'})
            elif key == 'n' and self.manual:
                self.manual = False
                self.manual_pending_until = time.monotonic() + 1.5
                self.publish('log', {'message': 'Operator DONE received; recording 1.5 seconds more.'})
            elif key == 'x':
                self.state = 'EXITED'
                self.options = []
                self.step = None
                self.running = False
                self._emit_state()

    def _verdict(self):
        return 'good' if len(self.faces) == 3 and all(s['verdict'] == 'good' for s in self.faces) else 'defective'

    def _snap(self, face):
        pattern = [(0.42, 0.63, 0.58), (0.58, 0.998, 0.47), (0.83, 0.79, 0.56), (0.65, 0.72, 0.78)]
        score = pattern[(self.cube_n - 1) % len(pattern)][face - 1]
        attempts = self.snap_retries.get(face, 0) + 1
        self.snap_retries[face] = attempts
        if 0.825 <= score < 0.975 and attempts == 1:
            verdict = 'uncertain'
        elif 0.825 <= score < 0.975 and attempts >= 2:
            verdict = 'defective'  # still uncertain => defective by policy
        elif score >= .975:
            verdict = 'defective'
        else:
            verdict = 'good'
        snap = {'face': face, 'attempt': attempts, 'verdict': verdict,
                'score': score, 'low': 0.825, 'high': 0.975,
                'find': {'ok': True, 'reasons': [], 'ms': 38 + face*7},
                'images': {}}
        self.latest_snap = snap
        if face <= len(self.faces):
            self.faces[face-1] = snap
        else:
            self.faces.append(snap)
        self.publish('snap', snap)
        return verdict == 'uncertain'

    def _finish(self, duration):
        step = self.step
        self.steps.append({'name': step, 'kind': self.kind,
                           'duration_s': round(self.t, 2), 'ok': True,
                           'reason': 'DONE', 'chunks': int(self.t * .8), 'attempt': 1})
        self.publish('step_finished', {'step': step, 'ok': True, 'reason': 'DONE',
                                       'duration_s': round(self.t, 2), 'chunks': int(self.t*.8),
                                       'rejected': 0, 'max_correction': 0.1, 'latency_ms': 230})
        if step.startswith('SNAP_'):
            face = int(step[-1])
            retry = self._snap(face)
            if retry:
                # One extra snap, no more. The existing state is SNAP_{face}.
                self.started = time.monotonic()
                self.t = 0
                self.publish('log', {'message': f'Uncertain face {face}: re-snap once.'})
                self.publish('step_started', {'step': step, 'kind': 'SNAP', 'prompt': '',
                                               'attempt': 2, 'timeout_s': self.timeout, 'rules_source': 'simulated'})
                return
        if step == 'DECIDE':
            self.publish('decision', {'cube_n': self.cube_n, 'verdict': self._verdict(),
                                      'reason': 'Good only when all three faces are seen and good',
                                      'faces': [dict(s) for s in self.faces]})
        if self.index == len(STEP_DEFS)-1:
            if not self.cancelled:
                verdict = self._verdict()
                self.counts['total'] += 1
                self.counts['good' if verdict == 'good' else 'defective'] += 1
                cycle = round(time.monotonic() - self.cycle_started, 2)
                self.cycle_times.append(cycle)
                self._save_cube(verdict)
                self.publish('run_done', {'run_dir': str(self.run_dir / self.run_name),
                                          'cubes': self.counts['total'], 'counts': self.counts.copy(),
                                          'cycle_time': round(sum(self.cycle_times)/len(self.cycle_times), 1)})
                self.cube_n += 1
            self.state = 'STANDBY'
            self.step = None
            self.kind = None
            self.prompt = ''
            self.options = ['g', 'r', 'x']
            self.manual = False
            self._emit_state()
        else:
            self._enter(self.index+1)

    def _save_cube(self, verdict):
        try:
            root = self.run_dir / self.run_name / f'cube_{self.cube_n:03d}'
            root.mkdir(parents=True, exist_ok=True)
            report = {'cube': self.cube_n, 'started': datetime.now().isoformat(timespec='seconds'),
                      'finished': datetime.now().isoformat(timespec='seconds'),
                      'verdict': verdict, 'reason': 'All 3 faces good' if verdict == 'good' else 'Defect or uncertain face',
                      'steps': [dict(s) for s in self.steps],
                      'snaps': [dict(s) for s in self.faces]}
            (root / 'cube.json').write_text(json.dumps(report, indent=2))
            for index, st in enumerate(self.steps):
                if st['kind'] != 'VLA':
                    continue
                d = st['duration_s']
                rows = []
                for j in range(max(2, int(d*30))):
                    t = j/30
                    fraction = min(1, t/max(d,.1))
                    rows.append({'t': round(t,3),
                                 'trend': round(max(.04, (1-fraction)**2*6+abs(math.sin(t*3))*.26),3),
                                 'wobble': round(max(.025, (1-fraction)**2*2.9+abs(math.cos(t*2.1))*.11),3),
                                 'pose_err': round(max(.015,.45*(1-fraction)**2+abs(math.sin(t*2))*.035),3)})
                logfile = root / f'log_{index:02d}_{st["name"].lower()}.json'
                logfile.write_text(json.dumps({'step':st['name'], 'rows':rows,
                                               'auto_done_t':round(d*.93,2), 't_operator':None}))
        except Exception:
            # Simulation persistence failure must not interrupt the station.
            pass

    def _run(self):
        while self.running:
            with self.lock:
                if self.state not in ('STANDBY','EXITED') and self.index >= 0:
                    elapsed = time.monotonic() - self.started
                    duration = STEP_DEFS[self.index][2]
                    progress = min(1, elapsed / duration)
                    self.t = elapsed
                    self.pose_err = max(.015, .45 * (1-progress)**2 + abs(math.sin(elapsed*2))*.035)
                    self.trend = max(.04, (1-progress)**2*6 + abs(math.sin(elapsed*3))*.26)
                    self.wobble = max(.025, (1-progress)**2*2.9 + abs(math.cos(elapsed*2.1))*.11)
                    self.still = max(0, elapsed-duration*.72)
                    self.flat = max(0, int((elapsed-duration*.6)*2))
                    self.publish('step_status', {'t': round(elapsed,2), 'pose_err': round(self.pose_err,3),
                                                'trend_mm_s': round(self.trend,2), 'wobble_mm': round(self.wobble,2),
                                                'plan_flat': self.flat, 'still_s': round(self.still,2),
                                                'rules_say': 'DONE' if progress>.93 else 'MOVING',
                                                'manual': self.manual})
                    if elapsed >= duration and not self.manual and (not self.manual_pending_until or time.monotonic() >= self.manual_pending_until):
                        self._finish(duration)
            time.sleep(.25)
