import cv2
import numpy as np
import os
import sys
from collections import deque
from scipy.interpolate import CubicSpline

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.sectors.release import ReleaseSector
from core.sectors.tracking import TrackingSector
from core.sectors.physics import PhysicsSector
from vision.pose_analyzer import PoseAnalyzer
from vision.detector import BallDetector

class DeliveryState:
    WAITING = "WAITING"
    RELEASED = "RELEASED"
    TRACKING = "TRACKING"
    FINISHED = "FINISHED"

class SurgicalTracker:
    """
    MODULAR SECTOR ENGINE.
    Coordinates between Release, Tracking, and Physics sectors.
    """
    def __init__(self, ball_model_path, pose_model_path, device="cpu"):
        # 1. Initialize Vision Engines (Heavy)
        self.pose_analyzer = PoseAnalyzer(model_path=pose_model_path, device=device)
        self.ball_detector = BallDetector(model_path=ball_model_path, conf=0.15)

        # 2. Initialize Sectors (Logic)
        self.release_sector = ReleaseSector(self.pose_analyzer)
        self.tracking_sector = TrackingSector(self.ball_detector)
        self.physics_sector = PhysicsSector()

        # 3. Global State
        self.state = DeliveryState.WAITING
        self.trajectory = []
        self.smooth_path = []
        self.live_smooth_path = []               # Live incremental smooth curve during tracking
        self.handoff_frames = 0
        self.max_handoff = 12 
        self.cooldown_frames = 0
        self.max_cooldown = 150  # 5 seconds at 30 fps
        
        self.frame_buffer = deque(maxlen=5)     # 5-frame rolling buffer
        self.release_delay = 0                   # FIX: Detect immediately at release
        self.frames_since_release = 0            # Counter
        
        self.tracking_misses = 0                 # FIX: Allow tolerance for lost frames
        self.max_tracking_misses = 5             # Allow 5 missed frames before giving up

    def reset(self):
        self.state = DeliveryState.WAITING
        self.trajectory = []
        self.smooth_path = []
        self.live_smooth_path = []
        self.handoff_frames = 0
        self.frames_since_release = 0
        self.tracking_misses = 0
        self.frame_buffer.clear()
        self.release_sector.reset()
        self.tracking_sector.reset()
        self.physics_sector.reset()

    def set_bowling_config(self, bowler_side=None, wicket_side=None):
        self.release_sector.set_config(bowler_side, wicket_side)

    def process_frame(self, frame, roi_data):
        """
        Main Sectorized Pipeline Loop with Safety Guards.
        """
        status = {"state": self.state, "release_pt": self.release_sector.latest_peak}

        # 0. Cooldown Handler (Stop pose/ball scanning for a few seconds)
        if self.cooldown_frames > 0:
            self.cooldown_frames -= 1
            status["state"] = "COOLDOWN"
            self._draw_physics(frame)
            return frame, status

        try:
            # --- SECTOR 1: RELEASE DETECTION ---
            if self.state == DeliveryState.WAITING:
                is_release, peak_pt, shld_pt, frame = self.release_sector.process(frame, roi_data.get("bowler_roi"))
                if is_release:
                    self.state = DeliveryState.RELEASED
                    self.handoff_frames = 0
                    self.ball_detector.prev_frame_crop = None  # FIX: Reset motion state for first corridor detection
                    print(f"!!! [AI TRIGGER] MODULAR RELEASE DETECTED at {peak_pt} !!!")

            # --- SECTOR 2: HANDOFF & TRACKING (WITH DELAYED DETECTION) ---
            elif self.state == DeliveryState.RELEASED:
                self.frame_buffer.append(frame.copy())
                self.handoff_frames += 1
                self.frames_since_release += 1

                # --- DUAL ROI FIRST DETECTION ---
                h, w = frame.shape[:2]
                corridor = None
                secondary_corridor = None

                if self.release_sector.latest_peak and self.release_sector.latest_shoulder:
                    px, py = self.release_sector.latest_peak
                    sx, sy = self.release_sector.latest_shoulder
                    mid_x = w // 2
                    # --- STRAIGHT SQUARE/RECTANGULAR CORRIDOR ---
                    # Create a generous rectangular green box around the shoulder and wrist
                    pad_x = 100
                    pad_top = 120    # Extend high up for the ball arc
                    pad_bot = 40     # Extend slightly below shoulder
                    
                    x1 = max(0, min(sx, px) - pad_x)
                    y1 = max(0, min(sy, py) - pad_top)
                    x2 = min(w, max(sx, px) + pad_x)
                    y2 = min(h, max(sy, py) + pad_bot)
                    
                    corridor = (int(x1), int(y1), int(x2), int(y2))
                    
                    # Diagnostic logging
                    print(f"  > Square Corridor: ({x1},{y1}) to ({x2},{y2})")

                # Process the straight square green box
                ball, frame = self.tracking_sector.process_corridor(
                    frame, corridor=corridor, is_release_handoff=True)

                # --- VERTICAL SHOULDER LINE GATE ---
                # (Visual line removed as requested)
                if sx is not None:
                    if ball and ball["center"][0] < sx:
                        print(f"  > [SHOULDER GATE] Rejected: cx={ball['center'][0]} is left of shoulder x={sx}")
                        ball = None

                if ball:
                    # WRIST ANCHORING: If this is the very first detection, 
                    # start the trajectory exactly at the bowler's wrist coordinate.
                    if not self.trajectory and self.release_sector.latest_peak:
                        self.trajectory.append(self.release_sector.latest_peak)
                    
                    self.trajectory.append(ball["center"])
                    # GATHER UP TO 3 POINTS FOR KALMAN WARMUP
                    warmup_count = len(self.trajectory)
                    if warmup_count >= 3:
                        self.state = DeliveryState.TRACKING
                        self.frames_since_release = 0
                        print(f"--- [AI SUCCESS] Kalman Warmup Complete ({warmup_count}/3). Transitioning to Tracking. ---")
                    else:
                        print(f"  > [WARMUP] Gathering Kalman data: {warmup_count}/3 pts")
                else:
                    # FLEXIBLE CONSTRAINT: If we already have 1 or 2 detections, but miss now,
                    # the ball likely left the corridor. Transition to TRACKING anyway to let Kalman predict.
                    if len(self.trajectory) >= 1:
                        print(f"--- [AI HANDOFF] Ball left corridor with {len(self.trajectory)} warmup pts. Transitioning. ---")
                        self.state = DeliveryState.TRACKING
                        self.frames_since_release = 0
                    elif self.handoff_frames >= self.max_handoff:
                        print(f"--- [AI TIMEOUT] No Ball in Corridor after {self.max_handoff} frames. Resetting. ---")
                        self.reset()


            elif self.state == DeliveryState.TRACKING:
                last_pt = self.trajectory[-1]
                # FIX: Widen search box to 200px (adaptive is for stabilization)
                box = (last_pt[0], last_pt[1], 200) 
                ball, frame = self.tracking_sector.process(frame, search_box=box)
                
                if ball:
                    # Trajectory gating: only accept downward movement or slight correction
                    if ball["center"][1] > last_pt[1] - 8:
                        self.trajectory.append(ball["center"])
                        self.tracking_misses = 0 # Reset on success
                        # Recompute live smooth path every time a point is added
                        self.live_smooth_path = self._compute_live_smooth()
                    else:
                        print("--- [AI FINISH] Ball went backwards. Finalizing. ---")
                        self._finalize()
                else:
                    # FIX: Momentum Guard (Allow 5 missed frames before giving up)
                    self.tracking_misses += 1
                    if self.tracking_misses >= self.max_tracking_misses:
                        print(f"--- [AI TIMEOUT] Tracking lost for {self.max_tracking_misses} frames. Finalizing. ---")
                        self._finalize()
                    else:
                        cv2.putText(frame, f"TRACKING LOST - {self.max_tracking_misses - self.tracking_misses} TRIES LEFT",
                                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)

        except Exception as e:
            print(f"--- [ERROR] Tracker processing failed: {e} ---")
            status["error"] = str(e)

        # --- SECTOR 3: PHYSICS VISUALIZATION ---
        self._draw_physics(frame)

        return frame, status

    def _draw_physics(self, frame):
        """
        Consolidated drawing for trajectory, parabolas, and bounce points.
        Persists through all states as long as data exists.
        """
        # 1. Draw raw trajectory dots (small, subtle — the smooth curve is the hero)
        for p in self.trajectory:
            cv2.circle(frame, p, 2, (255, 220, 0), -1)
            
        # 2a. During TRACKING: draw live incremental smooth curve
        if self.state == DeliveryState.TRACKING and self.live_smooth_path:
            for i in range(1, len(self.live_smooth_path)):
                cv2.line(frame, self.live_smooth_path[i-1], self.live_smooth_path[i], (0, 230, 255), 2)

        # 2b. After finalization: draw high-quality fitted path
        if self.smooth_path:
            for i in range(1, len(self.smooth_path)):
                cv2.line(frame, self.smooth_path[i-1], self.smooth_path[i], (0, 255, 255), 3) 
            
            if self.physics_sector.bounce_point:
                cv2.circle(frame, self.physics_sector.bounce_point, 8, (255, 0, 255), 2) 

    def _compute_live_smooth(self):
        """
        Incrementally fits a parametric cubic spline to the current trajectory
        for live display during TRACKING state. Requires ≥4 points.
        Returns a list of (int, int) pixel tuples.
        """
        if len(self.trajectory) < 4:
            # Not enough data yet — return straight line segments between raw points
            return [(int(p[0]), int(p[1])) for p in self.trajectory]

        pts = np.array(self.trajectory, dtype=np.float64)
        # Parameterize by cumulative arc-length for best spline behaviour
        diffs = np.diff(pts, axis=0)
        distances = np.sqrt((diffs ** 2).sum(axis=1))
        t = np.concatenate([[0], np.cumsum(distances)])

        if t[-1] == 0:
            return [(int(p[0]), int(p[1])) for p in self.trajectory]

        try:
            cs_x = CubicSpline(t, pts[:, 0], bc_type='natural')
            cs_y = CubicSpline(t, pts[:, 1], bc_type='natural')
            num_samples = max(len(self.trajectory) * 8, 60)
            t_fine = np.linspace(t[0], t[-1], num_samples)
            return [(int(cs_x(ti)), int(cs_y(ti))) for ti in t_fine]
        except Exception:
            return [(int(p[0]), int(p[1])) for p in self.trajectory]

    def _finalize(self):
        """
        Triggered when tracking stops. Delegates to Physics Sector.
        """
        self.cooldown_frames = self.max_cooldown
        self.live_smooth_path = []  # Clear live curve; final curve takes over
        if len(self.trajectory) >= 3:
            print(f"--- [AI FINISH] fitting smooth trajectory to {len(self.trajectory)} pts ---")
            self.state = DeliveryState.FINISHED
            self.smooth_path = self.physics_sector.fit(self.trajectory)
        else:
            print(f"--- [AI FAIL] Insufficient data. Showing dots only. ---")
            self.state = DeliveryState.FINISHED # Keep state to prevent jumpy UI
