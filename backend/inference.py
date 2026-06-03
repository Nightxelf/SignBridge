import json
import os
import numpy as np
import torch
import cv2

from backend.models.lstm_model import SignLSTM
from backend.mediapipe_handler import MediaPipeHandler

MODEL_PATH = os.environ.get("SIGNBRIDGE_MODEL_PATH", "backend/checkpoints/latest.pt")
LABEL_MAP_PATH = os.environ.get("SIGNBRIDGE_LABEL_MAP", "backend/label_map.json")
_model = None
_mp_handler = None
_label_map = {0: "hello", 1: "thank_you", 2: "yes", 3: "no"}


def load_label_map():
    global _label_map
    if os.path.exists(LABEL_MAP_PATH):
        try:
            with open(LABEL_MAP_PATH, "r", encoding="utf-8") as fp:
                loaded = json.load(fp)
            if not isinstance(loaded, dict):
                return

            if all(isinstance(v, str) for v in loaded.values()) and all(str(k).isdigit() for k in loaded.keys()):
                _label_map = {int(k): v for k, v in loaded.items()}
                return

            if all(isinstance(v, int) for v in loaded.values()):
                _label_map = {int(v): str(k) for k, v in loaded.items()}
                return

            try:
                _label_map = {int(k): v for k, v in loaded.items()}
                return
            except Exception:
                pass
        except Exception:
            pass


def _ensure_handlers():
    global _model, _mp_handler
    if _mp_handler is None:
        try:
            _mp_handler = MediaPipeHandler()
        except Exception:
            _mp_handler = None
    if _model is None:
        load_label_map()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SignLSTM(input_size=63, hidden_size=128, num_layers=2, num_classes=max(1, len(_label_map)))
        if os.path.exists(MODEL_PATH):
            try:
                state = torch.load(MODEL_PATH, map_location=device)
                model.load_state_dict(state)
            except Exception:
                pass
        model.to(device)
        model.eval()
        _model = model


def predict_from_landmarks(landmarks_sequence):
    """landmarks_sequence: list of frames, each a flat list of 63 floats"""
    _ensure_handlers()
    if _model is None:
        return "unknown", 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = np.array(landmarks_sequence, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    x = torch.from_numpy(x).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = _model(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        idx = int(probs.argmax())
        conf = float(probs[idx])
        label = _label_map.get(idx, "unknown")
    return label, conf


def predict_from_image(image_bytes: bytes):
    """Run MediaPipe on an image and predict from extracted landmarks (single-frame fallback)
    Expects image bytes (jpg/png).
    """
    _ensure_handlers()
    if _mp_handler is None:
        return "unknown", 0.0
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return "unknown", 0.0, None
    result = _mp_handler.extract_from_bgr(img)
    if not result:
        return "no_hand", 0.0, None
    coords = result.get("coords")
    bbox = result.get("bbox")
    if not coords:
        return "no_hand", 0.0, bbox
    label, conf = predict_from_landmarks([coords])
    return label, conf, bbox
