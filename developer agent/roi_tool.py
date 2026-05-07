import cv2
import os
import sys

# Ensure the vision module is found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vision.roi_selector import ROISelector
from core.config import VIDEO_PATH

def main():
    # Setup
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {VIDEO_PATH}")
        return

    # Initialize Selector
    selector = ROISelector(window_name="ROI Tool - Surgical DRS")
    cv2.namedWindow(selector.window_name)
    cv2.setMouseCallback(selector.window_name, selector.mouse_callback)

    print("\n--- ROI TOOL CONTROLS ---")
    print("LEFT CLICK: Add coordinate")
    print("'c': Confirm Polygon")
    print("'r': Reset ROI")
    print("'m': Toggle Mask Mode")
    print("'ESC': Exit")
    print("--------------------------\n")

    mask_mode = False

    while True:
        # If confirmed, read continuously. Else, keep first frame for selection.
        if not selector.is_confirmed:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        ret, frame = cap.read()
        if not ret:
            break

        # Processing
        display_frame = frame.copy()
        display_frame = selector.draw_roi(display_frame)

        if mask_mode and selector.is_confirmed:
            display_frame = selector.apply_roi(display_frame)

        # UI Overlay
        status = "MODE: SELECTION" if not selector.is_confirmed else "MODE: PLAYBACK"
        cv2.putText(display_frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if mask_mode:
            cv2.putText(display_frame, "MASK: ON", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(selector.window_name, display_frame)

        key = cv2.waitKey(30) & 0xFF

        if key == 27: # ESC
            break
        elif key == ord('c'): # Confirm
            if len(selector.points) >= 3:
                selector.is_confirmed = True
                print("ROI Confirmed.")
            else:
                print("Error: Select at least 3 points for a polygon.")
        elif key == ord('r'): # Reset
            selector.reset()
            mask_mode = False
        elif key == ord('m'): # Toggle Mask
            mask_mode = not mask_mode

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
