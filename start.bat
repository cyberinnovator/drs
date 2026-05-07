@echo off
echo Starting the DRS Developer Agent...
call venv\Scripts\activate.bat
cd "developer agent"
python main.py
pause
