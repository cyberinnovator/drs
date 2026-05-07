@echo off
echo Setting up the DRS Developer Agent Project...
python -m venv venv
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r req.txt
echo Setup complete. You can now run the project using start.bat
pause
