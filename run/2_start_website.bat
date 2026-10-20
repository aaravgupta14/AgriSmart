@echo off
cd /d "%~dp0.."
echo Starting AgriSmart at http://127.0.0.1:5000  (Ctrl+C to stop)
.venv\Scripts\python.exe -m webapp.app
