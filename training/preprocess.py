"""Dataset preprocessing utilities.

This file contains placeholders for converting video datasets (WLASL or similar) into
per-frame landmark sequences using MediaPipe. Implement dataset-specific IO in practice.
"""
import os
import cv2
import numpy as np
from backend.mediapipe_handler import MediaPipeHandler


def extract_landmarks_from_video(video_path, out_dir=None):
    mp = MediaPipeHandler()
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        coords = mp.extract_from_bgr(frame)
        frames.append(coords)
    cap.release()
    return frames


if __name__ == "__main__":
    print("Preprocess utilities. Call functions from training scripts.")
