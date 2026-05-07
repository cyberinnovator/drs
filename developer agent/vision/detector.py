import cv2
import numpy as np
from ultralytics import YOLO
from core import config

class BallDetector:
    def __init__(self, model_path, conf=0.25):
        self.model = YOLO(model_path)
        self.conf = conf
        self.roi_points = None
        self.roi_mask = None
        self.roi_bbox = None # (x, y, w, h)
        self.last_processed_crop = None
        self.prev_frame_crop = None   # for frame differencing
        self.last_motion_mask = None  # saved for fallback
        self.debug_stages = {}        # NEW: for visual debugging
        
        # Modular settings from config
        self.mode = config.ACTIVE_DETECTOR  # "YOLO" or "HSV"
        self.ball_color = config.BALL_COLOR # "RED", "WHITE", "PINK"

    def set_roi(self, points, frame_shape=(640, 640)):
        """Points should be in original/target resolution."""
        if points is None or len(points) < 2:
            return
        
        fh, fw = frame_shape[:2]
        self.roi_points = np.array(points, dtype=np.int32)
        
        # Calculate Bounding Box for Cropped Inference (with clipping)
        if len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            bx, by = min(x1, x2), min(y1, y2)
            bw, bh = abs(x2 - x1), abs(y2 - y1)
            
            # Project to top if close to top for ribbons, or just keep square
            if by < 50: by = 0
            
        else:
            bx, by, bw, bh = cv2.boundingRect(self.roi_points)
            
        # --- CLIPPING SAFETY ---
        x1_clip = max(0, bx)
        y1_clip = max(0, by)
        x2_clip = min(fw, bx + bw)
        y2_clip = min(fh, by + bh)
        
        # Ensure minimum size (don't crash on tiny ROIs)
        if (x2_clip - x1_clip) < 5 or (y2_clip - y1_clip) < 5:
            self.roi_bbox = (0, 0, fw, fh)
            self.mask_poly = np.array([[0,0], [fw,0], [fw,fh], [0,fh]], dtype=np.int32)
        else:
            self.roi_bbox = (x1_clip, y1_clip, x2_clip - x1_clip, y2_clip - y1_clip)
            if len(points) == 2:
                # Rectangular mask for box search
                self.mask_poly = np.array([
                    [x1_clip, y1_clip], [x2_clip, y1_clip], 
                    [x2_clip, y2_clip], [x1_clip, y2_clip]
                ], dtype=np.int32)
            else:
                self.mask_poly = self.roi_points
            
        # (roi_bbox is now ready for use in detect())

    def detect(self, frame):
        """
        Runs detection specifically within the ROI.
        Routes to YOLO or HSV based on configuration.
        """
        if self.roi_points is None:
            # Fallback: Detect on whole frame if no ROI
            results = self.model(frame, conf=self.conf, verbose=False)
            return self._parse_results(results)

        # 1. CROP & MASK
        x, y, w, h = self.roi_bbox
        crop = frame[y:y+h, x:x+w].copy()
        
        # Localize points relative to crop for spatial gating later
        local_pts = self.mask_poly - [x, y]
        local_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(local_mask, [local_pts], 255)
        
        # Strict background masking for classical logic
        input_masked = cv2.bitwise_and(crop, crop, mask=local_mask)
        
        self.debug_stages["1_crop"] = crop.copy()
        self.debug_stages["2_roi_masked"] = input_masked.copy()

        # Route to active detector
        if config.ACTIVE_DETECTOR == "HSV":
            detections = self._detect_hsv(crop, x, y)
        else:
            detections = self._detect_yolo(crop, x, y)
            
        # --- STRICT SPATIAL GATE ---
        # Final confirmation: Only return detections inside the green area
        filtered = []
        for d in detections:
            cx, cy = d["center"]
            if cv2.pointPolygonTest(self.mask_poly, (cx, cy), False) >= 0:
                filtered.append(d)
        
        return filtered

    def _detect_yolo(self, crop, offset_x, offset_y):
        """YOLO Inference Engine (RGB)"""
        # YOLO needs the natural box image (the crop) without artificial sharp black edges
        input_yolo = crop.copy()
        self.last_processed_crop = input_yolo
        self.debug_stages["5_yolo_input_rgb"] = input_yolo.copy()

        results = self.model(input_yolo, conf=self.conf, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                coords = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                gx1, gy1 = coords[0] + offset_x, coords[1] + offset_y
                gx2, gy2 = coords[2] + offset_x, coords[3] + offset_y
                cx, cy = int((gx1 + gx2) / 2), int((gy1 + gy2) / 2)
                
                detections.append({
                    "box": [int(gx1), int(gy1), int(gx2), int(gy2)],
                    "conf": conf,
                    "center": (cx, cy)
                })
        return detections

    def _detect_hsv(self, crop, offset_x, offset_y):
        """Temporal HSV Engine: Color + Motion Confirmation"""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # 1. COLOR MASKING
        if config.BALL_COLOR == "WHITE":
            lower = np.array([0, 0, 150])
            upper = np.array([180, 80, 255])
            color_mask = cv2.inRange(hsv, lower, upper)
        elif config.BALL_COLOR == "PINK":
            lower = np.array([140, 50, 50])
            upper = np.array([175, 255, 255])
            color_mask = cv2.inRange(hsv, lower, upper)
        elif config.BALL_COLOR == "REFERENCE":
            lower = np.array([10, 44, 192])
            upper = np.array([125, 114, 255])
            color_mask = cv2.inRange(hsv, lower, upper)
        else: # Default RED
            lower1 = np.array([0, 50, 30])
            upper1 = np.array([20, 255, 255])
            lower2 = np.array([155, 50, 30])
            upper2 = np.array([180, 255, 255])
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            color_mask = cv2.bitwise_or(mask1, mask2)

        # 2. MOTION MASKING (Temporal Confirmation)
        # Only moving colored objects are candidates for the ball.
        motion_mask = np.ones_like(color_mask) * 255 # Assume all pixels move on first frame
        if self.prev_frame_crop is not None and self.prev_frame_crop.shape == crop.shape:
            # Grayscale diff
            prev_gray = cv2.cvtColor(self.prev_frame_crop, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(curr_gray, prev_gray)
            _, motion_mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
            
            # Dilate motion to capture edges
            motion_mask = cv2.dilate(motion_mask, np.ones((5,5), np.uint8), iterations=1)

        # 3. COMBINED LOGIC
        final_mask = cv2.bitwise_and(color_mask, motion_mask)
        
        # Cleanup
        kernel = np.ones((3,3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        
        # Debug stages
        self.debug_stages["3_hsv_mask"] = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
        self.debug_stages["4_motion_mask"] = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
        self.debug_stages["5_combined_mask"] = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
        
        self.last_processed_crop = cv2.bitwise_and(crop, crop, mask=final_mask)

        # Find Blobs
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        
        if config.BALL_COLOR == "REFERENCE":
            # --- REFERENCE REPO LOGIC: Find LARGEST blob and use MOMENTS for center ---
            if len(contours) > 0:
                largest_cnt = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_cnt)
                
                if area > 10: # Minimum noise floor
                    M = cv2.moments(largest_cnt)
                    if M["m00"] != 0:
                        cx_local = int(M["m10"] / M["m00"])
                        cy_local = int(M["m01"] / M["m00"])
                        
                        # Map to global
                        gx, gy = cx_local + offset_x, cy_local + offset_y
                        bx, by, bw, bh = cv2.boundingRect(largest_cnt)
                        gx1, gy1 = bx + offset_x, by + offset_y
                        gx2, gy2 = gx1 + bw, gy1 + bh
                        
                        detections.append({
                            "box": [int(gx1), int(gy1), int(gx2), int(gy2)],
                            "conf": 1.0, # Target found
                            "center": (gx, gy)
                        })
        else:
            # --- ORIGINAL SURGICAL LOGIC: Multiple candidates + Shape gating ---
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if not (15 < area < 1000): continue
                
                # 1. Circularity Check: 4*pi*Area / Perimeter^2
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0: continue
                circularity = (4 * np.pi * area) / (perimeter * perimeter)
                if circularity < 0.6: continue # Reject jagged shapes
                
                # 2. Aspect Ratio Check
                bx, by, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = float(bw) / (bh + 1e-6)
                if not (0.6 < aspect_ratio < 1.6): continue
                
                # Success - map to global
                gx1, gy1 = bx + offset_x, by + offset_y
                gx2, gy2 = bx + bw + offset_x, by + bh + offset_y
                cx, cy = gx1 + bw // 2, gy1 + bh // 2
                
                detections.append({
                    "box": [int(gx1), int(gy1), int(gx2), int(gy2)],
                    "conf": circularity, # Use circularity as a proxy for 'confidence'
                    "center": (cx, cy)
                })
            
        return detections

    def detect_motion_fallback(self, frame):
        """
        Fallback ball detection using classical contour analysis on motion mask.
        Used when YOLO/HSV returns 0 detections inside the corridor.
        """
        # (This remains as a secondary safety layer)
        # Note: self.last_motion_mask is updated in TrackingSector if needed, 
        # or we could calculate it here if frame diffing is enabled.
        return [] # Placeholder - motion logic can be integrated if needed

    def _parse_results(self, results):
        detections = []
        for r in results:
            for box in r.boxes:
                coords = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cx, cy = int((coords[0] + coords[2]) / 2), int((coords[1] + coords[3]) / 2)
                detections.append({
                    "box": [int(c) for c in coords],
                    "conf": conf,
                    "center": (cx, cy)
                })
        return detections
