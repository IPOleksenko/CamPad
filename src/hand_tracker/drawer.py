import cv2

def draw_landmark_ids(image, hands_landmarks):
    for hand in hands_landmarks:
        for id, x, y in hand:
            cv2.circle(image, (x, y), 4, (0, 255, 0), cv2.FILLED)
            cv2.putText(image, str(id), (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
