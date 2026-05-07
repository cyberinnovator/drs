# DRS Developer Agent

This repository contains the DRS (Decision Review System) Developer Agent. It uses a computer vision pipeline with YOLO models to track a cricket ball in a video and visualizes its trajectory.

## Prerequisites

- Python 3.8+
- Git

## Installation

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/cyberinnovator/drs.git
   cd drs
   ```

2. **Create and activate a virtual environment**:
   - **Windows**:
     ```cmd
     python -m venv venv
     venv\Scripts\activate
     ```
   - **Linux/macOS**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install the dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r req.txt
   ```

## Running the Application

1. Ensure your virtual environment is activated.
2. Navigate to the core project directory:
   ```bash
   cd "developer agent"
   ```
3. Run the Flask application:
   ```bash
   python main.py
   ```

Once started, the Flask application will be available at `http://127.0.0.1:5000`. You can open this URL in your web browser to view the tracking interface.

## Project Structure

- `developer agent/`: The core source code containing the computer vision logic and the Flask backend.
- `req.txt`: Python package requirements needed to run the project.
- `latest.pt`: The YOLOv8 model for ball detection.
- `yolov8n-pose.pt`: The YOLOv8 pose model for biomechanical trigger (release) detection.
- `clip_6 - Trim.mp4`: A sample cricket video that the system processes.
