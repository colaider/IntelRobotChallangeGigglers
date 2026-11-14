#!/usr/bin/env bash
# Station check: benchmark every model on CPU / GPU / NPU, concurrent load, control-loop jitter,
# power and temperature -> results/station/report.md. Run from the repo root:
#   bash scripts/station_run.sh                 # defaults
#   bash scripts/station_run.sh --runs 200      # extra args go to vision/station_check.py
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

if [ ! -f results/models/models.json ]; then
  echo "No results/models/models.json - train first:"
  echo "  python vision/build_views.py && python vision/train_views.py --int8 && python vision/train_face_classifier.py"
  exit 1
fi
[ -r /sys/class/powercap/intel-rapl:0/energy_uj ] || \
  echo "(power will be n/a - for energy numbers: sudo chmod -R a+r /sys/class/powercap/intel-rapl:0)"

python vision/station_check.py "$@"
echo
echo "Report: results/station/report.md   (raw: station.json, bench.csv)"
