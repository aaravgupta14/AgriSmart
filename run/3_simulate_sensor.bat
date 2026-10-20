@echo off
cd /d "%~dp0.."
.venv\Scripts\python.exe hardware\bridge\serial_bridge.py --simulate --interval 10
