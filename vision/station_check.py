"""Station check for the Intel Core Ultra laptop: one run -> the OpenVINO optimisation table for the slide.

What it measures (everything is written to results/station/report.md + .json + .csv):
  1. system:      CPU, kernel, OpenVINO version, devices (CPU / GPU / NPU) with their full names
  2. per model x device x precision (FP16 / INT8 exports found next to each other):
                  compiles? (unsupported ops are reported, not fatal), first compile time (cold) and
                  with the model cache (warm), latency p50 / p95, FPS, energy per inference (RAPL)
  3. concurrent:  Anomalib on --anomaly-device + classifier on --classifier-device + a proxy load on
                  --policy-device (stand-in for the VLA policy) all at once -> latency alone vs together
  4. control loop: a 100 Hz loop on the CPU (stand-in for robot control) -> timing jitter idle vs under
                  the full AI load ("robot control stays reliable", slide 6 item 5)
  5. thermal:     max CPU package temperature idle / after load
  6. recommendation: device per model (fastest that works; NPU preferred for Anomalib when it is close)

Models come from results/models/models.json (train_views.py + train_face_classifier.py); extra IR
folders can be added with --models. Devices that are missing are skipped. Power needs read access to
/sys/class/powercap (sudo, or: sudo chmod -R a+r /sys/class/powercap); otherwise it is reported as n/a.

    python vision/station_check.py                         # everything, defaults
    python vision/station_check.py --runs 200 --load-seconds 20
"""

import argparse
import csv
import json
import os
import platform
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import openvino as ov

from ov_utils import CACHE_DIR, available, compile_with_fallback, core, device_name, dummy_inputs, is_available

RAPL = Path("/sys/class/powercap/intel-rapl:0/energy_uj")


# ---------------------------------------------------------------- system probes
def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def energy_j() -> float | None:
    try:
        return int(RAPL.read_text()) / 1e6
    except (OSError, ValueError):
        return None


def max_temp_c() -> float | None:
    temps = []
    for z in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            kind = (z / "type").read_text().strip()
            t = int((z / "temp").read_text()) / 1000
            if kind in ("x86_pkg_temp", "TCPU", "acpitz") or "pkg" in kind or "cpu" in kind.lower():
                temps.append(t)
        except (OSError, ValueError):
            continue
    return max(temps) if temps else None


def idle_power(seconds: float = 3.0) -> float | None:
    e0 = energy_j()
    if e0 is None:
        return None
    time.sleep(seconds)
    e1 = energy_j()
    return (e1 - e0) / seconds if e1 is not None and e1 >= e0 else None


# ---------------------------------------------------------------- models
def find_models(registry: Path, extra: list[Path]) -> list[dict]:
    out = []
    if registry.exists():
        reg = json.loads(registry.read_text())["cameras"]
        for cam, e in reg.items():
            entries = [("all", e.get("single"))] + list((e.get("per_face") or {}).items())
            for face, m in entries:
                if not m:
                    continue
                fp16 = Path(m["xml"])
                out.append({"name": f"{cam}/{face}", "role": "anomaly", "precision": "fp16", "xml": fp16})
                int8 = Path(str(fp16).replace("/fp16/", "/int8/"))
                if int8.exists():
                    out.append({"name": f"{cam}/{face}", "role": "anomaly", "precision": "int8", "xml": int8})
            if e.get("classifier"):
                out.append({"name": f"{cam}/classifier", "role": "classifier", "precision": "fp16",
                            "xml": Path(e["classifier"]["xml"])})
    for d in extra:
        for x in sorted(d.rglob("model.xml")):
            prec = next((p for p in ("int8", "fp16", "fp32") if f"/{p}/" in str(x)), "?")
            out.append({"name": str(x.parent.relative_to(d)), "role": "anomaly", "precision": prec, "xml": x})
    return out


# ---------------------------------------------------------------- measurements
def bench(xml: Path, dev: str, runs: int) -> dict:
    """Compile on exactly `dev` (no fallback) and time raw inference."""
    t = time.perf_counter()
    r = compile_with_fallback(xml, dev, fallback=False)
    if r.compiled is None:
        return {"ok": False, "error": r.tried[-1][1] if r.tried else "?"}
    first = time.perf_counter() - t
    t = time.perf_counter()
    r2 = compile_with_fallback(xml, dev, fallback=False)  # second compile: served from the model cache
    warm = time.perf_counter() - t
    c = r2.compiled or r.compiled
    x = dummy_inputs(core().read_model(str(xml)))
    for _ in range(5):
        c(x)
    e0, times = energy_j(), []
    for _ in range(runs):
        t = time.perf_counter()
        c(x)
        times.append((time.perf_counter() - t) * 1000)
    e1 = energy_j()
    joules = (e1 - e0) / runs if e0 is not None and e1 is not None and e1 >= e0 else None
    return {"ok": True, "compile_s": round(first, 2), "compile_cached_s": round(warm, 2),
            "p50_ms": round(statistics.median(times), 2), "p95_ms": round(float(np.percentile(times, 95)), 2),
            "fps": round(1000 / statistics.mean(times), 1), "mj_per_inf": round(joules * 1000, 1) if joules else None}


def control_loop(stop: threading.Event, hz: float, out: list):
    """Stand-in for the robot control loop: wake every 1/hz s, record how late each tick is (ms)."""
    period, nxt = 1.0 / hz, time.perf_counter()
    while not stop.is_set():
        nxt += period
        delay = nxt - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        out.append(max(0.0, (time.perf_counter() - nxt) * 1000))


def jitter(samples: list) -> dict:
    if not samples:
        return {}
    return {"p50_ms": round(statistics.median(samples), 2), "p99_ms": round(float(np.percentile(samples, 99)), 2),
            "max_ms": round(max(samples), 2), "ticks": len(samples)}


def infer_loop(compiled, x, stop: threading.Event, out: list):
    while not stop.is_set():
        t = time.perf_counter()
        compiled(x)
        out.append((time.perf_counter() - t) * 1000)


def concurrent(jobs: dict, seconds: float, hz: float) -> dict:
    """jobs: name -> (compiled, inputs). Run all together + the control loop; also each alone."""
    res = {"alone": {}, "together": {}}
    for name, (c, x) in jobs.items():
        stop, lat = threading.Event(), []
        th = threading.Thread(target=infer_loop, args=(c, x, stop, lat))
        th.start(); time.sleep(min(3.0, seconds / 2)); stop.set(); th.join()
        res["alone"][name] = round(statistics.median(lat), 2) if lat else None
    base = []
    stop = threading.Event()
    th = threading.Thread(target=control_loop, args=(stop, hz, base))
    th.start(); time.sleep(3.0); stop.set(); th.join()
    res["control_idle"] = jitter(base)

    stop, lats, ctrl = threading.Event(), {n: [] for n in jobs}, []
    threads = [threading.Thread(target=infer_loop, args=(c, x, stop, lats[n])) for n, (c, x) in jobs.items()]
    threads.append(threading.Thread(target=control_loop, args=(stop, hz, ctrl)))
    t0, e0, temp0 = time.perf_counter(), energy_j(), max_temp_c()
    for th in threads:
        th.start()
    time.sleep(seconds)
    stop.set()
    for th in threads:
        th.join()
    dt, e1 = time.perf_counter() - t0, energy_j()
    for n, lat in lats.items():
        res["together"][n] = {"p50_ms": round(statistics.median(lat), 2) if lat else None,
                              "fps": round(len(lat) / dt, 1)}
    res["control_load"] = jitter(ctrl)
    res["package_w"] = round((e1 - e0) / dt, 1) if e0 is not None and e1 is not None and e1 >= e0 else None
    res["temp_c"] = {"before": temp0, "after": max_temp_c()}
    return res


# ---------------------------------------------------------------- report
def recommend(rows: list[dict]) -> dict:
    best = {}
    for role in ("anomaly", "classifier"):
        cand = [r for r in rows if r["role"] == role and r.get("ok")]
        if not cand:
            continue
        by_dev = {}
        for r in cand:
            by_dev.setdefault(r["device"], []).append(r["p50_ms"])
        med = {d: statistics.median(v) for d, v in by_dev.items()}
        fastest = min(med, key=med.get)
        pick = fastest
        if role == "anomaly" and "NPU" in med and med["NPU"] <= 1.5 * med[fastest]:
            pick = "NPU"  # close enough: NPU = lowest power, frees the GPU for the VLA policy
        best[role] = {"device": pick, "median_p50_ms": {d: round(v, 2) for d, v in med.items()}}
    return best


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registry", type=Path, default=Path("results/models/models.json"))
    p.add_argument("--models", type=Path, nargs="*", default=[], help="extra folders with exported model.xml")
    p.add_argument("--devices", nargs="+", default=["CPU", "GPU", "NPU"])
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--anomaly-device", default="NPU")
    p.add_argument("--classifier-device", default="GPU")
    p.add_argument("--policy-device", default="GPU", help="proxy load standing in for the VLA policy")
    p.add_argument("--load-seconds", type=float, default=10.0)
    p.add_argument("--control-hz", type=float, default=100.0)
    p.add_argument("--out", type=Path, default=Path("results/station"))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    info = {"cpu": cpu_name(), "os": f"{platform.system()} {platform.release()}", "python": platform.python_version(),
            "openvino": ov.get_version(), "devices": {d: device_name(d) for d in available()},
            "rapl": energy_j() is not None, "cache_dir": str(CACHE_DIR)}
    info["idle_w"] = idle_power()
    info["idle_temp_c"] = max_temp_c()
    print(json.dumps(info, indent=2))

    models = find_models(a.registry, a.models)
    if not models:
        raise SystemExit(f"no models: run train_views.py first or pass --models <folder>")
    devices = [d for d in a.devices if is_available(d)]
    print(f"\n{len(models)} models x devices {devices}  (missing: {[d for d in a.devices if d not in devices]})")
    rows = []
    for m in models:
        for d in devices:
            r = bench(m["xml"], d, a.runs)
            row = {**{k: str(v) for k, v in m.items()}, "device": d, **r}
            rows.append(row)
            print(f"  {m['name']:22s} {m['precision']:5s} {d:4s} " + (
                f"p50 {r['p50_ms']:7.2f} ms  p95 {r['p95_ms']:7.2f}  {r['fps']:7.1f} fps  compile {r['compile_s']:.1f}s"
                f" (cached {r['compile_cached_s']:.1f}s)" + (f"  {r['mj_per_inf']} mJ" if r.get("mj_per_inf") else "")
                if r["ok"] else f"FAILED: {r['error']}"))

    # concurrent run with the planned device map (each falls back if its device is missing)
    anomaly = next((m for m in models if m["role"] == "anomaly" and m["precision"] == "fp16"), None)
    classifier = next((m for m in models if m["role"] == "classifier"), None)
    heavy = max((m for m in models if m["role"] == "anomaly"), key=lambda m: m["xml"].with_suffix(".bin").stat().st_size)
    jobs, devmap = {}, {}
    for name, m, dev in (("anomaly", anomaly, a.anomaly_device), ("classifier", classifier, a.classifier_device),
                         ("policy_proxy", heavy, a.policy_device)):
        if m is None:
            continue
        r = compile_with_fallback(m["xml"], dev)
        if r.compiled is not None:
            jobs[name] = (r.compiled, dummy_inputs(core().read_model(str(m["xml"]))))
            devmap[name] = {"asked": dev, "runs_on": r.device, "model": m["name"]}
    print(f"\nconcurrent load {a.load_seconds:.0f}s: {devmap}")
    conc = concurrent(jobs, a.load_seconds, a.control_hz)
    conc["device_map"] = devmap
    print(json.dumps(conc, indent=2))
    rec = recommend(rows)

    # ---- write outputs
    (a.out / "station.json").write_text(json.dumps({"system": info, "bench": rows, "concurrent": conc,
                                                    "recommendation": rec}, indent=2, default=str))
    keys = sorted({k for r in rows for k in r})
    with (a.out / "bench.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    md = ["# Station check\n", f"- CPU: {info['cpu']}", f"- OS: {info['os']}, OpenVINO {info['openvino']}",
          "- Devices: " + ", ".join(f"{k} ({v})" for k, v in info["devices"].items()),
          f"- Idle package power: {str(round(info['idle_w'], 1)) + ' W' if info['idle_w'] is not None else 'n/a (RAPL not readable: sudo)'}, "
          f"idle temp: {info['idle_temp_c'] if info['idle_temp_c'] is not None else 'n/a'} °C\n",
          "## Models × devices\n",
          "| model | prec | device | p50 ms | p95 ms | FPS | compile s (cached) | mJ / inference |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("ok"):
            md.append(f"| {r['name']} | {r['precision']} | {r['device']} | {r['p50_ms']} | {r['p95_ms']} | {r['fps']} | "
                      f"{r['compile_s']} ({r['compile_cached_s']}) | {r.get('mj_per_inf') or 'n/a'} |")
        else:
            md.append(f"| {r['name']} | {r['precision']} | {r['device']} | failed: {r['error']} | | | | |")
    md += ["\n## Everything at once (planned device map)\n",
           "| job | asked | runs on | alone p50 ms | together p50 ms | together FPS |", "|---|---|---|---|---|---|"]
    for n, dm in devmap.items():
        t = conc["together"].get(n, {})
        md.append(f"| {n} ({dm['model']}) | {dm['asked']} | {dm['runs_on']} | {conc['alone'].get(n)} | "
                  f"{t.get('p50_ms')} | {t.get('fps')} |")

    ci, cl = conc.get("control_idle", {}), conc.get("control_load", {})
    md += [f"\nControl loop {a.control_hz:.0f} Hz on the CPU (robot stand-in), lateness: idle p99 {ci.get('p99_ms')} ms / "
           f"max {ci.get('max_ms')} ms → under full AI load p99 {cl.get('p99_ms')} ms / max {cl.get('max_ms')} ms.",
           f"Package power under load: {str(conc.get('package_w')) + ' W' if conc.get('package_w') else 'n/a'}; temperature {conc['temp_c']['before']} → "
           f"{conc['temp_c']['after']} °C.\n", "## Recommended device map\n"]
    for role, r in rec.items():
        md.append(f"- **{role}** → **{r['device']}** (median p50 per device: {r['median_p50_ms']})")
    md.append("\nRaw data: station.json, bench.csv.")
    (a.out / "report.md").write_text("\n".join(md) + "\n")
    print(f"\nReport: {a.out / 'report.md'}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    main()
