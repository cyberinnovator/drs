import os

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH = os.path.join(os.path.dirname(BASE_DIR), "clip_6 - Trim.mp4")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# --- VIDEO CONFIG ---
IMG_SIZE = 640
FPS = 30

# --- DETECTION ---
ROOT_DIR = os.path.dirname(BASE_DIR)
YOLO_MODEL = os.path.join(ROOT_DIR,"latest.pt")
POSE_MODEL = os.path.join(ROOT_DIR, "yolov8n-pose.pt")
CONF_THRESHOLD = 0.4
SPORTS_BALL_CLS = 0

ACTIVE_DETECTOR = "YOLO"  # Options: "YOLO", "HSV"
BALL_COLOR = "RED"   # Options: "RED", "WHITE", "PINK", "REFERENCE"
