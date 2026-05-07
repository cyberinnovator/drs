import numpy as np
import cv2

class ReleaseSector:
    """
    Handles the detection of the bowler's hand peak and the release event.
    Supports manual Bowler Arm and Wicket Side selection.
    """
    def __init__(self, pose_analyzer):
        self.pose_analyzer = pose_analyzer
        self.latest_peak = None
        self.latest_shoulder = None  # Cache the shoulder at the peak event
        self.arm_side = "right"   # "left" or "right"
        self.wicket_side = "right" # "left" or "right" (Over/Around)
        self.manual_arm = False

    def reset(self):
        self.latest_peak = None
        self.latest_shoulder = None
        self.pose_analyzer.prev_wrist = None
        self.pose_analyzer.wrist_y_history.clear()

    def set_config(self, bowler_side=None, wicket_side=None):
        if bowler_side:
            self.arm_side = bowler_side
            self.manual_arm = True
        if wicket_side:
            self.wicket_side = wicket_side
        print(f"--- [CONFIG] ReleaseSector updated: Arm={self.arm_side}, Wicket={self.wicket_side} ---")

    def process(self, frame, ribbon_roi):
        """
        Scans for bowler pose and triggers release on hand peak.
        
        COCO Arm Index Reference:
          Right arm (bowler's right hand): shoulder=6, elbow=8, wrist=10
          Left  arm (bowler's left  hand): shoulder=5, elbow=7, wrist=9
        
        Wicket Side gating:
          wicket_side == "left"  → only scan persons in LEFT  half (cx < mid_x)
          wicket_side == "right" → only scan persons in RIGHT half (cx > mid_x)
        """
        h, w = frame.shape[:2]
        mid_x = w // 2
        kpts = None

        # --- Draw debug overlay FIRST (always visible) ---
        # 1. Vertical dividing line at frame center
        overlay_col = (0, 255, 255) if self.wicket_side == "right" else (255, 255, 0)
        active_label = "ACTIVE ZONE"
        
        # Shade the INACTIVE half (dim it slightly)
        inactive_region = frame.copy()
        if self.wicket_side == "right":
            # Active = right half, dim left half
            inactive_region[:, :mid_x] = (inactive_region[:, :mid_x].astype(float) * 0.5).astype(np.uint8)
        else:
            # Active = left half, dim right half
            inactive_region[:, mid_x:] = (inactive_region[:, mid_x:].astype(float) * 0.5).astype(np.uint8)
        frame[:] = inactive_region

        # Draw dividing line
        cv2.line(frame, (mid_x, 0), (mid_x, h), (255, 255, 255), 1)

        # 2. Config HUD (top-left corner)
        arm_label = f"ARM: {'RIGHT (R wrist=kp10)' if self.arm_side == 'right' else 'LEFT (L wrist=kp9)'}"
        side_label = f"SIDE: {self.wicket_side.upper()} HALF"
        cv2.putText(frame, arm_label,  (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, side_label, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

        # --- POSE DETECTION ---
        if ribbon_roi and len(ribbon_roi) >= 2:
            # SURGICAL MODE: scan only inside calibrated ROI
            kpts = self.pose_analyzer.get_bowler_keypoints(frame, ribbon_roi, side=self.arm_side)
        
        if kpts is None:
            # FALLBACK: Full-frame scan, strict wicket-side gate
            try:
                results = self.pose_analyzer.model(frame, verbose=False, device=self.pose_analyzer.device)
                if results and len(results) > 0 and results[0].keypoints is not None:
                    keypoints_data = results[0].keypoints.data.cpu().numpy()
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    
                    for i, box in enumerate(boxes):
                        cx = (box[0] + box[2]) / 2
                        # STRICT: only accept person on the correct half
                        if (self.wicket_side == "left"  and cx < mid_x) or \
                           (self.wicket_side == "right" and cx > mid_x):
                            kpts = keypoints_data[i]
                            break
                    # NOTE: NO fallback to first person — wicket side must be respected
            except Exception as e:
                print(f"  > [WARN] Full-frame pose scan failed: {e}")
                return False, None, None, frame

        if kpts is None:
            return False, None, None, frame

        # Auto-detect arm ONLY if not manually set
        if not self.manual_arm:
            # COCO: kp10=right wrist, kp9=left wrist. Higher confidence = bowling arm
            self.arm_side = "right" if kpts[10][2] > kpts[9][2] else "left"

        # Draw skeleton for SELECTED arm only
        frame = self.pose_analyzer.draw_skeleton(frame, kpts, side=self.arm_side)
        
        # Highlight the active wrist with a clear marker
        wrist_idx = 10 if self.arm_side == "right" else 9  # COCO: 10=R wrist, 9=L wrist
        wr = kpts[wrist_idx]
        if wr[2] > 0.15:
            wx, wy = int(wr[0]), int(wr[1])
            cv2.circle(frame, (wx, wy), 12, (0, 0, 255), 2)    # Red outer ring
            cv2.circle(frame, (wx, wy), 4,  (0, 255, 0), -1)   # Green fill
            cv2.putText(frame, f"{'R' if self.arm_side == 'right' else 'L'} WRIST (kp{wrist_idx})",
                        (wx + 14, wy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        is_release, wrist_pos, shoulder_pos = self.pose_analyzer.detect_release(kpts, side=self.arm_side)
        
        # FIX: Only update latest_peak on the ACTUAL release trigger
        # Also add sanity guard for wrist y > 10 (reject 0-edge detections)
        if is_release and wrist_pos and wrist_pos[1] > 10:
            self.latest_peak = wrist_pos
            self.latest_shoulder = shoulder_pos  # Save shoulder at peak
        
        return is_release, self.latest_peak, self.latest_shoulder, frame
