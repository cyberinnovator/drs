import numpy as np
import cv2

class TrackingSector:
    """
    Handles frame-by-frame ball tracking using YOLO and Kalman Filter.
    Uses ADAPTIVE search box size based on Kalman covariance (research-backed).
    """
    def __init__(self, ball_detector):
        self.ball_detector = ball_detector
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix  = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix   = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        # High process noise for velocity (index 2,3) to allow sudden bounce response
        self.kf.processNoiseCov = np.array([
            [1e-2, 0,    0,    0],
            [0,    1e-2, 0,    0],
            [0,    0,    5.0,  0],  
            [0,    0,    0,    5.0]
        ], np.float32)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.is_initialized = False
        self.last_ball_pos = None
        self.last_miss = False

    def reset(self):
        self.is_initialized = False
        self.last_ball_pos = None
        self.last_miss = False
        self.kf.statePost = np.zeros((4, 1), np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

    def _init_kalman(self, pos):
        self.kf.statePost = np.array(
            [[np.float32(pos[0])], [np.float32(pos[1])], [0], [0]], np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.is_initialized = True

    def _get_adaptive_box_size(self, base_size=45):
        """
        Research-backed adaptive search box based on Kalman covariance.
          - High uncertainty (post-bounce/lost)  → box expands up to 120px
          - Low uncertainty (confident tracking) → box shrinks to ~45px
        Ref: Adaptive Kalman Filtering for sports tracking
        """
        if not self.is_initialized:
            return 80 # Larger initial search box for handoff
        px_var = float(self.kf.errorCovPost[0, 0])
        py_var = float(self.kf.errorCovPost[1, 1])
        uncertainty = (px_var + py_var) ** 0.5
        # Scale: base 45px + up to 75px extra from uncertainty
        adaptive = int(base_size + min(uncertainty * 2, 75))
        return max(40, min(adaptive, 120))

    def process(self, frame, roi_pts=None, search_box=None, is_release_handoff=False):
        """
        Predicts next ball position and runs YOLO in the search region.
        
        Visualizations:
          🟡 Yellow box   — Kalman search region (adaptive size shown as r=Xpx)
          🟢 Green box    — Detected ball confirmed by YOLO
          🟠 Orange dot   — Kalman raw prediction point
        """
        if not roi_pts and not search_box:
            return None, frame

        fh, fw = frame.shape[:2]

        # 1. Kalman Prediction
        prediction = self.kf.predict()
        pred_x, pred_y = int(prediction[0]), int(prediction[1])

        # 2. Adaptive search region
        if search_box:
            cx, cy = search_box[0], search_box[1]
            # At release: fixed large box (ball position unknown)
            # During tracking: covariance-adaptive (smaller as confidence grows)
            r = search_box[2] if is_release_handoff else self._get_adaptive_box_size()
        else:
            r = self._get_adaptive_box_size()
            # If the previous frame was a miss, center the box on the last known position
            # to prevent the Kalman filter from overshooting/drifting too far (User's request)
            if self.last_miss and self.last_ball_pos is not None:
                cx, cy = self.last_ball_pos
            else:
                cx, cy = pred_x, pred_y

        sx1 = max(0, cx - r)
        sy1 = max(0, cy - r)
        sx2 = min(fw, cx + r)
        sy2 = min(fh, cy + r)
        self.ball_detector.set_roi([(sx1, sy1), (sx2, sy2)], frame_shape=frame.shape)

        # YOLO Inference — ultra-low conf at release to catch fast-leaving ball
        orig_conf = self.ball_detector.conf
        if is_release_handoff:
            self.ball_detector.conf = 0.05
        detections = self.ball_detector.detect(frame)
        self.ball_detector.conf = orig_conf
        
        # Draw YELLOW SEARCH BOX (always visible, shows adaptive size) AFTER detection
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 255, 255), 1)
        cv2.putText(frame, f"SEARCH r={r}px", (sx1 + 2, max(0, sy1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

        # --- DRAW AI VISION PIP ---
        self._draw_ai_vision(frame)

        best_ball = None
        max_allowed_shift = 90 # Generous box to catch fast upward bounce
        
        if detections:
            if not self.is_initialized:
                best_ball = detections[0]
                self._init_kalman(best_ball["center"])
            else:
                # Pick detection closest to Kalman prediction
                candidates = []
                for d in detections:
                    dist = np.linalg.norm(np.array(d["center"]) - np.array([pred_x, pred_y]))
                    if dist < max_allowed_shift:
                        candidates.append((d, dist))
                
                if candidates:
                    candidates.sort(key=lambda x: x[1])
                    best_ball = candidates[0][0]
                    self.last_ball_pos = best_ball["center"]
                    self.last_miss = False
                    # Update Kalman filter
                    meas = np.array([[np.float32(best_ball["center"][0])],
                                      [np.float32(best_ball["center"][1])]], np.float32)
                    self.kf.correct(meas)
                else:
                    self.last_miss = True
                    # OCCLUSION GUARD: No valid detection → trust prediction, skip correction
                    pass 
        else:
            self.last_miss = True

        # 5. Draw GREEN BOX around detected ball
        if best_ball:
            bx1, by1, bx2, by2 = best_ball["box"]
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.putText(frame, f"BALL {best_ball['conf']:.2f}", (bx1, max(0, by1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(frame, best_ball["center"], 4, (0, 255, 0), -1)

        # 6. Orange Kalman prediction dot
        if self.is_initialized:
            cv2.circle(frame, (pred_x, pred_y), 5, (0, 165, 255), 2)

        return best_ball, frame

    def process_corridor(self, frame, corridor=None, is_release_handoff=False, draw_overlay=True):
        """
        First Detection Corridor — searches for ball within a precise geometric zone.
        Supports both rectangular (x1, y1, x2, y2) and polygonal [(x,y),...] corridors.
        """
        if corridor is None:
            return self.process(frame, is_release_handoff=is_release_handoff)

        fh, fw = frame.shape[:2]
        is_poly = isinstance(corridor, list) or (isinstance(corridor, np.ndarray) and corridor.ndim == 2)

        if is_poly:
            # Polygonal ROI: calculate bounding box for YOLO crop, then mask inside
            roi_points = np.array(corridor, dtype=np.int32)
            bx, by, bw, bh = cv2.boundingRect(roi_points)
        else:
            # Rectangular ROI
            cx1, cy1, cx2, cy2 = corridor
            cx1 = max(0, cx1);  cy1 = max(0, cy1)
            cx2 = min(fw, cx2); cy2 = min(fh, cy2)

            # Ensure valid corridor
            if cx2 <= cx1 or cy2 <= cy1:
                return self.process(frame, is_release_handoff=is_release_handoff)

            roi_points = [(cx1, cy1), (cx2, cy1), (cx2, cy2), (cx1, cy2)]

        # Run YOLO inside the corridor (polygonal or rectangular) FIRST
        self.ball_detector.set_roi(roi_points, frame_shape=frame.shape)
        orig_conf = self.ball_detector.conf
        self.ball_detector.conf = 0.05  # Ultra-low confidence at release
        detections = self.ball_detector.detect(frame)
        self.ball_detector.conf = orig_conf

        if draw_overlay:
            if is_poly:
                # Draw translucent shaded polygon for visual confirmation
                overlay = frame.copy()
                cv2.fillPoly(overlay, [roi_points], (0, 255, 0))
                cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
                cv2.polylines(frame, [roi_points], True, (0, 255, 0), 2)
                cv2.putText(frame, "ANGLED BIOMECH CORRIDOR", (bx, max(20, by - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                # Draw the CORRIDOR on screen (prominent visual)
                cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)       # Green border
                cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 100, 0), 1)       # Dark fill line
                # Mid-line marker
                cv2.line(frame, (cx1, cy2), (cx2, cy2), (255, 255, 0), 1)          # Yellow mid line
                cv2.putText(frame, "BALL CORRIDOR", (cx1 + 4, cy1 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, "MID LINE", (cx1 + 4, cy2 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)

        # --- DRAW AI VISION PIP ---
        self._draw_ai_vision(frame)

        # Kalman prediction (for initializing if ball found)
        prediction = self.kf.predict()
        pred_x, pred_y = int(prediction[0]), int(prediction[1])

        best_ball = None
        max_allowed_shift = 90 # Generous box to catch fast upward bounce
        
        # If YOLO found nothing, try motion-based fallback
        if not detections:
            detections = self.ball_detector.detect_motion_fallback(frame)
            if detections:
                print("  > [FALLBACK] Motion-based contour detection succeeded.")

        if detections:
            if not self.is_initialized:
                best_ball = detections[0]
                self._init_kalman(best_ball["center"])
                self.last_ball_pos = best_ball["center"]
                self.last_miss = False
            else:
                # Pick detection closest to prediction
                candidates = []
                for d in detections:
                    dist = np.linalg.norm(np.array(d["center"]) - np.array([pred_x, pred_y]))
                    if dist < max_allowed_shift:
                        candidates.append((d, dist))
                
                if candidates:
                    best_ball = min(candidates, key=lambda x: x[1])[0]
                    self.last_ball_pos = best_ball["center"]
                    self.last_miss = False
                    meas = np.array([[np.float32(best_ball["center"][0])],
                                      [np.float32(best_ball["center"][1])]], np.float32)
                    self.kf.correct(meas)
                else:
                    self.last_miss = True
        else:
            if self.is_initialized:
                self.last_miss = True

        # Draw green ball box if found
        if best_ball:
            bx1, by1, bx2, by2 = best_ball["box"]
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.putText(frame, f"BALL {best_ball['conf']:.2f}", (bx1, max(0, by1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(frame, best_ball["center"], 5, (0, 255, 0), -1)

        return best_ball, frame

    def _draw_ai_vision(self, frame):
        """
        Draws the AI's internal preprocessed (CLAHE) view as a PiP window.
        """
        if hasattr(self.ball_detector, "last_processed_crop") and self.ball_detector.last_processed_crop is not None:
            pip_img = self.ball_detector.last_processed_crop
            fh, fw = frame.shape[:2]
            ph, pw = pip_img.shape[:2]
            
            # Place in the top-right corner
            margin = 10
            x_offset = fw - pw - margin
            y_offset = margin
            
            # Safe boundary check
            if x_offset > 0 and y_offset > 0 and y_offset + ph < fh and x_offset + pw < fw:
                frame[y_offset:y_offset+ph, x_offset:x_offset+pw] = pip_img
                cv2.rectangle(frame, (x_offset, y_offset), (x_offset+pw, y_offset+ph), (0, 0, 255), 2)
                cv2.putText(frame, "AI VISION (CLAHE)", (x_offset, max(10, y_offset - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
