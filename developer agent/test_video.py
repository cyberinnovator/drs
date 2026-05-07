import cv2
import os

VIDEO_PATH = r"c:\Users\HP\Desktop\drs\clip_6 - Trim.mp4"
print(f"Testing file at: {VIDEO_PATH}")
print(f"Exists: {os.path.exists(VIDEO_PATH)}")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("FAILED TO OPEN")
else:
    ret, frame = cap.read()
    print(f"Read First Frame: {ret}")
    if ret:
        print(f"Frame Shape: {frame.shape}")
cap.release()
