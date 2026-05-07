import os
import cv2
import base64
import json
import traceback
import torch
import numpy as np
import sys
from flask import Flask, render_template, jsonify, request

# Add parent directory to path to handle "developer agent" structure
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.config import VIDEO_PATH, IMG_SIZE, YOLO_MODEL, POSE_MODEL
from core.tracker import SurgicalTracker

app = Flask(__name__, template_folder='templates', static_folder='static')

# --- CONFIG & PERSISTENCE ---
ROI_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")

def load_roi():
    if os.path.exists(ROI_FILE):
        try:
            with open(ROI_FILE, 'r') as f:
                return json.load(f)
        except: pass
    return {"points": [], "pitching_zone": [], "bowler_roi": [], "confirmed": False}

def save_roi(data):
    with open(ROI_FILE, 'w') as f:
        json.dump(data, f)

# --- GLOBALS & STATE ---
cap = None
frame_idx = 0
roi_data = load_roi()
last_processed_frame = None  # FIX: Cache for EOF persistence

# Hardware Acceleration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- [HARDWARE] Initializing Surgical Pipeline on: {device.upper()} ---")

tracker = SurgicalTracker(
    ball_model_path=YOLO_MODEL, 
    pose_model_path=POSE_MODEL, 
    device=device
)

def init_cap():
    global cap, frame_idx
    if cap is not None:
        cap.release()
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"--- [CRITICAL] Failed to open video at {VIDEO_PATH} ---")
    else:
        print(f"--- Video Opened Successfully: {VIDEO_PATH} ---")
    frame_idx = 0

# Initialize Capture at startup
init_cap()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reset', methods=['POST'])
def reset():
    global frame_idx
    init_cap()
    tracker.reset()
    return jsonify({"success": True})

@app.route('/set_roi', methods=['POST'])
def set_roi():
    global roi_data
    try:
        data = request.json
        roi_type = data.get("type", "base")
        if roi_type == "base":
            roi_data["points"] = data.get("points", [])
        elif roi_type == "pitching":
            roi_data["pitching_zone"] = data.get("points", [])
        else: # Bowler ROI
            roi_data["bowler_roi"] = data.get("points", [])
            
        roi_data["confirmed"] = True
        save_roi(roi_data)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/clear_roi', methods=['POST'])
def clear_roi():
    global roi_data
    roi_data = {"points": [], "pitching_zone": [], "bowler_roi": [], "confirmed": False}
    save_roi(roi_data)
    return jsonify({"success": True})

@app.route('/step', methods=['GET'])
def step():
    global frame_idx, last_processed_frame
    if cap is None or not cap.isOpened():
        init_cap()
    
    ret, frame = cap.read()
    
    # If first frame fails, try to re-init once (Self-Healing)
    if not ret and frame_idx == 0:
        print("--- [VIDEO] First-frame Read Failed. Retrying Init... ---")
        init_cap()
        ret, frame = cap.read()

    if not ret:
        print(f"--- [VIDEO] EOF HIT at frame {frame_idx}. Showing last processed frame with trajectory. ---")
        # Ensure img_b64 is defined even in empty cases
        img_b64 = None
        
        if last_processed_frame is not None:
            _, buf = cv2.imencode('.jpg', last_processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_b64 = base64.b64encode(buf).decode('utf-8')
            return jsonify({"image": img_b64, "frame_idx": frame_idx, "eof": True})
            
        return jsonify({"eof": True, "image": None})
    
    frame_idx += 1
    return serve_frame(frame)

@app.route('/step_back', methods=['GET'])
def step_back():
    global frame_idx
    if cap is None or frame_idx <= 1:
        return jsonify({"error": "Cannot step back further", "frame_idx": frame_idx})
    
    # Seek to previous frame
    # frame_idx is current (1-indexed), so to get the previous frame, we seek to frame_idx - 2
    # Example: frame_idx=5. We just read frame 5. We want to go back to frame 4.
    # cv2.CAP_PROP_POS_FRAMES is 0-indexed. Frame 4 is index 3.
    # formula: index = frame_idx - 2
    new_idx = max(0, frame_idx - 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, new_idx)
    
    ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "Failed to read frame"})
    
    frame_idx -= 1
    return serve_frame(frame)

@app.route('/current_frame', methods=['GET'])
def current_frame():
    global frame_idx
    if cap is None or frame_idx <= 0:
        return jsonify({"error": "No frame loaded", "frame_idx": frame_idx})
    
    # Seek to current frame (cv2 POS_FRAMES is 0-indexed, frame_idx is 1-indexed)
    new_idx = max(0, frame_idx - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, new_idx)
    
    ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "Failed to read frame"})
    
    # Re-serve the frame with the latest tracker configuration
    # Note: We do NOT increment frame_idx here.
    return serve_frame(frame)

@app.route('/set_bowler_side', methods=['POST'])
def set_bowler_side():
    try:
        data = request.json
        side = data.get("side") # 'left' or 'right'
        if side in ['left', 'right']:
            tracker.set_bowling_config(bowler_side=side)
            return jsonify({"success": True})
    except: pass
    return jsonify({"success": False})

@app.route('/set_wicket_side', methods=['POST'])
def set_wicket_side():
    try:
        data = request.json
        side = data.get("side") # 'left' or 'right'
        if side in ['left', 'right']:
            tracker.set_bowling_config(wicket_side=side)
            return jsonify({"success": True})
    except: pass
    return jsonify({"success": False})

@app.route('/debug_pipeline')
def debug_pipeline():
    """
    Returns a composite grid image of all internal processing stages.
    Each panel is labelled and shown at a fixed size (200x150px per cell).
    """
    stages = tracker.ball_detector.debug_stages
    if not stages:
        # Return blank placeholder if no detection has run yet
        blank = np.zeros((150, 200, 3), dtype=np.uint8)
        cv2.putText(blank, "Step frame first", (5, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        _, buf = cv2.imencode('.jpg', blank)
        return jsonify({"image": base64.b64encode(buf).decode()})

    CELL_W, CELL_H = 200, 150
    labels = {
        "1_crop":          "1. Original Crop",
        "2_roi_masked":    "2. ROI Masked",
        "3_motion_diff":   "3. Motion Diff",
        "4_motion_mask":   "4. Motion Mask",
        "5_motion_only":   "5. Motion Isolated",
        "6_grayscale":     "6. Grayscale",
        "7_clahe_enhanced":"7. CLAHE to YOLO",
    }

    cells = []
    for key, label in labels.items():
        img = stages.get(key, np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8))
        img = cv2.resize(img, (CELL_W, CELL_H))
        # Add black title bar at top
        title_bar = np.zeros((20, CELL_W, 3), dtype=np.uint8)
        cv2.putText(title_bar, label, (4, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
        cell = np.vstack([title_bar, img])
        cells.append(cell)

    # Arrange in a grid (e.g. 2 rows x 4 cols to fit 7-8 stages)
    row1 = np.hstack(cells[:4])
    # Pad row2 if needed
    while len(cells[4:]) < 4:
        cells.append(np.zeros_like(cells[0]))
    row2 = np.hstack(cells[4:8])
    composite = np.vstack([row1, row2])

    # Header bar
    header = np.zeros((28, composite.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, "AI DETECTION PIPELINE - ALL STAGES", (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
    composite = np.vstack([header, composite])

    _, buf = cv2.imencode('.jpg', composite, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jsonify({"image": base64.b64encode(buf).decode()})

def serve_frame(frame):
    global frame_idx
    try:
        # Process for UI
        h, w = frame.shape[:2]
        scale = IMG_SIZE / max(h, w)
        display_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        
        # --- SURGICAL TRACKING & HANDOFF (Dynamic AI) ---
        display_frame, tracker_status = tracker.process_frame(display_frame, roi_data)

        # --- EXTRACT PIPELINE DEBUG STAGES (Base64) ---
        pipeline_stages = {}
        for key, img in tracker.ball_detector.debug_stages.items():
            if img is not None:
                _, s_buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                pipeline_stages[key] = base64.b64encode(s_buf).decode('utf-8')
        
        # --- DRAW STATIC MASKS (Visual Layout) ---
        if roi_data.get("confirmed"):
            # 1. PITCHING ZONE (Blue Highlight)
            if roi_data.get("pitching_zone"):
                p_pts = np.array(roi_data["pitching_zone"], np.int32)
                overlay = display_frame.copy()
                cv2.fillPoly(overlay, [p_pts], (180, 50, 50)) # Blue
                cv2.addWeighted(overlay, 0.4, display_frame, 0.6, 0, display_frame)
                cv2.polylines(display_frame, [p_pts], True, (255, 100, 100), 2)

            # 2. BASE ROI & MASKING
            if roi_data.get("points"):
                pts = np.array(roi_data["points"], np.int32)
                top_pts = np.array([[p[0], 0] for p in roi_data["points"]], np.int32)

                mask = np.zeros(display_frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)
                # Brightened Background (60% instead of 40%)
                background = (display_frame.astype(float) * 0.6).astype(np.uint8)
                
                # Apply Mask
                masked_content = cv2.bitwise_and(display_frame, display_frame, mask=mask)
                inverse_mask = cv2.bitwise_not(mask)
                background_layer = cv2.bitwise_and(background, background, mask=inverse_mask)
                display_frame = cv2.add(masked_content, background_layer)

                # Draw 3D Wireframe
                for i in range(len(roi_data["points"])):
                    p_base = tuple(roi_data["points"][i])
                    p_top = tuple(top_pts[i])
                    cv2.line(display_frame, p_base, p_top, (0, 255, 255), 2)
                cv2.polylines(display_frame, [pts], True, (0, 255, 255), 2)
                cv2.polylines(display_frame, [top_pts], True, (0, 255, 255), 2)

        # FIX: Cache result for EOF persistence
        global last_processed_frame
        last_processed_frame = display_frame.copy()

        _, buffer = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "frame_idx": frame_idx,
            "image": img_b64,
            "eof": False,
            "roi": roi_data,
            "state": tracker_status.get("state", "WAITING"),
            "pipeline": pipeline_stages # FIX: Embed processing stages
        })
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"--- SERVER ERROR ---\n{error_msg}")
        return jsonify({"error": str(e), "traceback": error_msg, "frame_idx": frame_idx}), 500

if __name__ == '__main__':
    # Initial Init
    init_cap()
    app.run(debug=True, port=8889, host='0.0.0.0')
