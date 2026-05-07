# DRS Developer Agent

This repository contains the DRS (Decision Review System) Developer Agent. It uses a computer vision pipeline with YOLO models to track a cricket ball in a video and visualizes its trajectory.

## Prerequisites

- Python 3.8+
- Git

## Installation

You can set up this project easily using the provided scripts.

### Windows

Double-click or run `setup.bat` in your terminal to create a virtual environment and install the required dependencies:

```cmd
setup.bat
```

### Linux / macOS

Make the setup script executable and run it:

```bash
chmod +x setup.sh
./setup.sh
```

## Running the Application

### Windows

Double-click or run `start.bat`:

```cmd
start.bat
```

### Linux / macOS

Make the start script executable and run it:

```bash
chmod +x start.sh
./start.sh
```

Once started, the Flask application will be available at `http://127.0.0.1:5000`. You can open this URL in your web browser to view the tracking interface.

## Project Structure

- `developer agent/`: The core source code containing the computer vision logic and the Flask backend.
- `req.txt`: Python package requirements needed to run the project.
- `latest.pt`: The YOLOv8 model for ball detection.
- `yolov8n-pose.pt`: The YOLOv8 pose model for biomechanical trigger (release) detection.
- `clip_6 - Trim.mp4`: A sample cricket video that the system processes.
