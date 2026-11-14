#!/usr/bin/env bash
# One-time setup on the station laptop (Ubuntu, Intel Core Ultra). Run from the repo root:
#   bash scripts/station_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python3.12}
command -v "$PY" >/dev/null || PY=python3
echo "== python: $($PY --version)"

if [ ! -d .venv ]; then
  if command -v uv >/dev/null; then uv venv --python 3.12 .venv; else "$PY" -m venv .venv; fi
fi
source .venv/bin/activate
PIP="pip"; command -v uv >/dev/null && PIP="uv pip"

# CPU-only torch first: the Intel GPU/NPU are used through OpenVINO, CUDA wheels would only waste GBs
$PIP install torch torchvision --index-url https://download.pytorch.org/whl/cpu
$PIP install -r requirements.txt

echo "== OpenVINO devices:"
python - <<'PY'
import openvino as ov
c = ov.Core()
for d in c.available_devices:
    print(f"   {d:6s} {c.get_property(d, 'FULL_DEVICE_NAME')}")
missing = [d for d in ("GPU", "NPU") if not any(a.startswith(d) for a in c.available_devices)]
if missing:
    print("   MISSING:", missing, "-> check drivers and that your user is in the 'render' group:")
    print("            groups | grep render || sudo usermod -aG render $USER   (then log out / in)")
    print("            ls -l /dev/accel/ /dev/dri/")
PY

if [ -r /sys/class/powercap/intel-rapl:0/energy_uj ]; then
  echo "== RAPL power counters: readable"
else
  echo "== RAPL power counters: NOT readable -> for power numbers run once:"
  echo "   sudo chmod -R a+r /sys/class/powercap/intel-rapl:0"
fi
echo "== done. Next: bash scripts/station_run.sh"
