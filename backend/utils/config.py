import os


class Settings:
    def __init__(self):
        self.host = os.getenv("SIGNBRIDGE_HOST", "0.0.0.0")
        self.port = int(os.getenv("SIGNBRIDGE_PORT", "8000"))
        self.model_path = os.getenv("SIGNBRIDGE_MODEL_PATH", "backend/checkpoints/latest.pt")
