#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
if [ -x "$HOME/intel-challenge/physical-ai-studio/application/backend/.venv/bin/python" ]; then
  PYTHON="$HOME/intel-challenge/physical-ai-studio/application/backend/.venv/bin/python"
fi
"$PYTHON" -c 'import fastapi, uvicorn' 2>/dev/null || "$PYTHON" -m pip install -r requirements.txt
exec "$PYTHON" launch.py
