#!/bin/bash
echo "Setting up the DRS Developer Agent Project..."
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r req.txt
echo "Setup complete. You can now run the project using ./start.sh"
