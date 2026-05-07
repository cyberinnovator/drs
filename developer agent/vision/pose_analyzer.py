import os
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

class PoseAnalyzer:
    def __init__(self, model_path="yolov8n-pose.pt", device="cpu"):
        # Search for model in various locations - Prioritize yolov8x-pose if it exists (legacy parity)
        possible_paths = [
            "yolov8x-pose.pt",
            "yolov8n-pose.pt",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "yolov8x-pose.pt"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "yolov8n-pose.pt"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "yolov8x-pose.pt"),
        ]
        
        target_path = model_path
        for p in possible_paths:
            if os.path.exists(p):
                target_path = p
                print(f"PoseAnalyzer: Found model at {target_path}")
                break

        self.model = YOLO(target_path)
        self.device = device
        self.model.to(device)
        
        self.prev_wrist = None
        self.wrist_y_history = deque(maxlen=5)  # Rolling 5-frame wrist Y for peak detection
        
        # COCO Pose Keypoint Map (Full 17-point schema)
        self.kp_map = {
            "nose": 0, "eye_l": 1, "eye_r": 2, "ear_l": 3, "ear_r": 4,
            "shoulder_l": 5, "shoulder_r": 6, "elbow_l": 7, "elbow_r": 8,
            "wrist_l": 9, "wrist_r": 10, "hip_l": 11, "hip_r": 12,
            "knee_l": 13, "knee_r": 14, "ankle_l": 15, "ankle_r": 16
        }

    def get_bowler_keypoints(self, frame, roi_points, side="right"):
        """
        Processes only the area within the Bowler ROI to find the bowler.
        Physically masks the INACTIVE arm to prevent detection interference.
        """
        if roi_points is None or len(roi_points) < 2:
            return None
            
        # 1. Surgical Crop & Ribbon Projection
        if len(roi_points) == 2:
            x1, x2 = roi_points[0][0], roi_points[1][0]
            bx, by = min(x1, x2), 0
            bw, bh = abs(x2 - x1), frame.shape[0]
            crop = frame[by:by+bh, bx:bx+bw].copy()
            x, y = bx, by
        else:
            pts = np.array(roi_points, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            crop = frame[y:y+h, x:x+w].copy()
        
        # 2. Pose Inference on Crop
        results = self.model(crop, verbose=False, device=self.device)
        if not results or len(results) == 0 or results[0].keypoints is None:
            return None
            
        if len(results[0].keypoints.data) == 0:
            return None
            
        keypoints = results[0].keypoints.data.cpu().numpy()[0] # Take first person
        
        # 3. MASK INACTIVE ARM (Physical Blanking)
        # COCO: Left (5, 7, 9), Right (6, 8, 10)
        if side == "left":
            # Zero out Right Arm (Shoulder=6, Elbow=8, Wrist=10)
            keypoints[6] = [0, 0, 0]
            keypoints[8] = [0, 0, 0]
            keypoints[10] = [0, 0, 0]
        elif side == "right":
            # Zero out Left Arm (Shoulder=5, Elbow=7, Wrist=9)
            keypoints[5] = [0, 0, 0]
            keypoints[7] = [0, 0, 0]
            keypoints[9] = [0, 0, 0]

        # 4. Map back to global coordinates
        global_kpts = keypoints.copy()
        for i in range(len(global_kpts)):
            if global_kpts[i][2] > 0.1: # Confidence
                global_kpts[i][0] += x
                global_kpts[i][1] += y
                
        return global_kpts

    def detect_release(self, kpts, side="right"):
        """
        Detects the ball release using WRIST PEAK DETECTION.

        Cricket bowling release mechanics:
          - Phase 1 (LOADING): Arm swings back, wrist_y INCREASES (arm going down)
          - Phase 2 (UPSWING): Arm comes over, wrist_y DECREASES  (arm going UP)
          - Phase 3 (RELEASE): Wrist reaches its HIGHEST point (wrist_y at minimum)
          - Phase 4 (FOLLOW): Arm follows through, wrist_y INCREASES again

        Release is confirmed when the wrist transitions from Phase 2 → Phase 4,
        i.e., the wrist Y STOPS decreasing and STARTS INCREASING after a clear rise.
        
        COCO: Right arm → kp[6](sh), kp[8](el), kp[10](wr)
              Left  arm → kp[5](sh), kp[7](el), kp[9](wr)
        """
        if kpts is None:
            self.wrist_y_history.clear()
            return False, None

        # Select correct arm keypoints
        if side == "right":
            sh, el, wr = kpts[6], kpts[8], kpts[10]
        else:
            sh, el, wr = kpts[5], kpts[7], kpts[9]
        # Require reasonable keypoint confidence
        if sh[2] < 0.15 or el[2] < 0.15 or wr[2] < 0.15:
            return False, None, None

        shoulder_pos = (int(sh[0]), int(sh[1]))
        wrist_pos = (int(wr[0]), int(wr[1]))
        wrist_y   = int(wr[1])

        # Store wrist Y history (lower Y = higher in frame = arm is going UP)
        self.wrist_y_history.append(wrist_y)
        self.prev_wrist = wrist_pos

        # Bio-mechanical constraint: Arm must be near-straight (160+ degrees)
        arm_angle = self._calculate_angle(sh, el, wr)
        is_straight = arm_angle > 160
        
        # Wrist must be physically higher (lower Y coordinate) than BOTH shoulder and elbow
        wrist_highest = wr[1] < sh[1] and wr[1] < el[1]

        # Trigger IMMEDIATELY when wrist reaches the top-arc position
        is_release = wrist_highest and is_straight

        return is_release, wrist_pos, shoulder_pos

    def _calculate_angle(self, a, b, c):
        ba = np.array([a[0] - b[0], a[1] - b[1]])
        bc = np.array([c[0] - b[0], c[1] - b[1]])
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def draw_skeleton(self, frame, kpts, side="right"):
        """
        Draws the skeletal map for DRS visualization based on selected arm.
        """
        if kpts is None: return frame
        
        # Professional Color Scheme
        L_COL = (255, 255, 0) # Left (Odd IDs)
        R_COL = (0, 255, 255) # Right (Even IDs)
        C_COL = (255, 0, 255) # Center
        
        # Full Skeleton Pairs (Simplified)
        all_bones = [
            (5, 7, L_COL), (7, 9, L_COL),  # Left Arm
            (6, 8, R_COL), (8, 10, R_COL), # Right Arm
            (5, 6, C_COL)                  # Shoulders
        ]
        
        # Filter bones based on side
        active_bones = []
        if side == "left":
            active_bones = [all_bones[0], all_bones[1], all_bones[4]]
            active_joints = [5, 7, 9, 6] # Keep both shoulders for context
        else:
            active_bones = [all_bones[2], all_bones[3], all_bones[4]]
            active_joints = [6, 8, 10, 5]

        # 1. Draw Bones (Lines)
        for i1, i2, color in active_bones:
            pt1 = kpts[i1]
            pt2 = kpts[i2]
            if pt1[2] > 0.15 and pt2[2] > 0.15:
                cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, 2)
        
        # 2. Draw Joints (Circles)
        for i in active_joints:
            pt = kpts[i]
            if pt[2] > 0.15:
                # Wrist gets a larger dot
                radius = 5 if i in [9, 10] else 3
                color = L_COL if i % 2 != 0 else R_COL
                if i in [5, 6]: color = C_COL
                cv2.circle(frame, (int(pt[0]), int(pt[1])), radius, color, -1)
        
        return frame
