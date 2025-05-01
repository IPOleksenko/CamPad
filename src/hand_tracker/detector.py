import mediapipe as mp
import cv2

class HandDetector:
    def __init__(self, max_hands=2):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.results = None

    def find_hands(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)
        return image

    def get_landmarks(self, image):
        landmarks_all = []
        handedness_all = []
        if self.results.multi_hand_landmarks:
            for hand_landmarks, hand_info in zip(
                self.results.multi_hand_landmarks,
                self.results.multi_handedness
            ):
                hand = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, _ = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    hand.append((id, cx, cy))
                landmarks_all.append(hand)
                handedness_all.append(hand_info.classification[0].label)
        return landmarks_all, handedness_all
