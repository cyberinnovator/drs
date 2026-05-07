import cv2
import numpy as np

class ROISelector:
    def __init__(self, window_name="ROI Selection"):
        self.points = []
        self.window_name = window_name
        self.is_confirmed = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.is_confirmed:
                self.points.append((x, y))
                print(f"Point added: ({x}, {y})")

    def draw_roi(self, frame):
        # Draw circles on each point
        for pt in self.points:
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)
        
        # Draw lines between points
        if len(self.points) > 1:
            pts = np.array(self.points, np.int32).reshape((-1, 1, 2))
            is_closed = self.is_confirmed
            cv2.polylines(frame, [pts], is_closed, (0, 255, 255), 2)
        
        return frame

    def get_mask(self, frame_shape):
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        if self.points:
            pts = np.array(self.points, np.int32)
            cv2.fillPoly(mask, [pts], 255)
        return mask

    def apply_roi(self, frame):
        mask = self.get_mask(frame.shape)
        return cv2.bitwise_and(frame, frame, mask=mask)

    def reset(self):
        self.points = []
        self.is_confirmed = False
        print("ROI Reset.")
