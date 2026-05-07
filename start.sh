#!/bin/bash
echo "Starting the DRS Developer Agent..."
source venv/bin/activate
cd "developer agent"
python3 main.py
