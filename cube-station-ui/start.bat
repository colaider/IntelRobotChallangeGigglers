@echo off
cd /d "%~dp0"
where py >nul 2>nul && (set "PY=py -3") || (set "PY=python")
%PY% -c "import fastapi,uvicorn" >nul 2>nul || %PY% -m pip install -r requirements.txt
%PY% launch.py
pause
