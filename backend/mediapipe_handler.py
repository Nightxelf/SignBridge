import cv2
import numpy as np

class MediaPipeHandler:
    def __init__(self, max_num_hands=1, min_detection_confidence=0.5):
        # Simplified fallback mode since Mediapipe is complex to set up
        self.enabled = False
        try:
            import mediapipe as mp
            if hasattr(mp, 'solutions'):
                self.hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=max_num_hands,
                    min_detection_confidence=min_detection_confidence
                )
                self.enabled = True
                self.api_type = 'solutions'
            else:
                print("Mediapipe solutions not available, using fallback detection")
                self.enabled = False
                self.api_type = 'fallback'
        except Exception as e:
            print(f"Mediapipe initialization failed: {e}, using fallback")
            self.enabled = False
            self.api_type = 'fallback'

    def extract_from_bgr(self, frame_bgr):
        if not self.enabled:
            return self._simple_fallback_detection(frame_bgr)
        
        if self.api_type == 'solutions':
            return self._extract_with_solutions(frame_bgr)
        else:
            return self._simple_fallback_detection(frame_bgr)
    
    def _simple_fallback_detection(self, frame_bgr):
        # Simple skin color detection as fallback
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour (likely the hand)
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 100:
            return None
        
        x, y, w, h = cv2.boundingRect(largest)
        h_img, w_img = frame_bgr.shape[:2]
        
        # Normalize to 0-1 range
        bbox = [float(x) / w_img, float(y) / h_img, float(x + w) / w_img, float(y + h) / h_img]
        
        # Create fake landmark coordinates (21 landmarks, 3 coords each = 63)
        coords = [0.5] * 63
        return {"coords": coords, "bbox": bbox}
    
    def _extract_with_solutions(self, frame_bgr):
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        if not results.multi_hand_landmarks:
            return None
        hand = results.multi_hand_landmarks[0]
        coords = []
        xs = []
        ys = []
        for lm in hand.landmark:
            xs.append(lm.x)
            ys.append(lm.y)
            coords.extend([lm.x, lm.y, lm.z])
        bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
        return {"coords": coords, "bbox": bbox}
