import numpy as np
from scipy.interpolate import CubicSpline

class PhysicsSector:
    """
    Handles trajectory fitting using Parametric Cubic Spline interpolation.
    Provides a perfectly smooth, broadcast-grade trajectory curve.
    """
    def __init__(self):
        self.smooth_path = []
        self.bounce_point = None

    def reset(self):
        self.smooth_path = []
        self.bounce_point = None

    def fit(self, trajectory):
        """
        Fits two independent parametric cubic splines to the trajectory segments.
        Splits at the bounce point to ensure a sharp, broadcast-quality impact angle.
        """
        if len(trajectory) < 3:
            return [(int(p[0]), int(p[1])) for p in trajectory]

        pts = np.array(trajectory, dtype=np.float64)
        y_vals = pts[:, 1]
        
        # 1. Detect bounce index (lowest point on screen = max Y)
        bounce_idx = np.argmax(y_vals)
        self.bounce_point = (int(pts[bounce_idx, 0]), int(pts[bounce_idx, 1]))

        # 2. Split trajectory into segments (overlap at bounce point)
        seg1 = pts[:bounce_idx + 1]
        seg2 = pts[bounce_idx:]

        full_smooth_path = []

        # 3. Process Segment 1 (Pre-bounce)
        if len(seg1) >= 3:
            full_smooth_path.extend(self._fit_segment(seg1))
        else:
            full_smooth_path.extend([(int(p[0]), int(p[1])) for p in seg1])

        # 4. Process Segment 2 (Post-bounce/Impact)
        if len(seg2) >= 3:
            full_smooth_path.extend(self._fit_segment(seg2))
        elif len(seg2) > 0:
            full_smooth_path.extend([(int(p[0]), int(p[1])) for p in seg2])

        self.smooth_path = full_smooth_path
        return self.smooth_path

    def _fit_segment(self, pts):
        """Helper to fit a single cubic spline to a point segment"""
        # Parameterize by cumulative arc length (t)
        diffs = np.diff(pts, axis=0)
        distances = np.sqrt((diffs ** 2).sum(axis=1))
        t = np.concatenate([[0], np.cumsum(distances)])
        
        if t[-1] == 0: # Case where points are overlapping or single
             return [(int(p[0]), int(p[1])) for p in pts]

        cs_x = CubicSpline(t, pts[:, 0], bc_type='natural')
        cs_y = CubicSpline(t, pts[:, 1], bc_type='natural')
        
        # Sample high resolution based on segment length
        num_samples = max(len(pts) * 10, 50)
        t_fine = np.linspace(t[0], t[-1], num_samples)
        
        path = []
        for time in t_fine:
            path.append((int(cs_x(time)), int(cs_y(time))))
        return path
