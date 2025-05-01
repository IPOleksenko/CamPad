import cv2
import sys
from hand_tracker.detector import HandDetector
from hand_tracker.drawer import draw_landmark_ids
from config import CAMERA_INDEX
from utils.window_utils import set_window_icon

def main():
    window_name = "CamPad"
    cap = cv2.VideoCapture(CAMERA_INDEX)
    detector = HandDetector()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)
    set_window_icon(window_name, "assets/icon.ico")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame = detector.find_hands(frame)
        landmarks_all, handedness_all = detector.get_landmarks(frame)

        for hand_landmarks, label in zip(landmarks_all, handedness_all):
            draw_landmark_ids(frame, [hand_landmarks])
            wrist_id, x, y = hand_landmarks[0]
            cv2.putText(frame, label, (x - 20, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 0, 0) if label == "Left" else (0, 0, 255), 2)

            # Output all landmarks values ​​to the image
            for lm_id, cx, cy in hand_landmarks:
                cv2.putText(frame, f"{lm_id}", (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1)
                cv2.putText(frame, f"({cx},{cy})", (cx + 10, cy + 10),
                            cv2.FONT_HERSHEY_PLAIN, 0.6, (150, 255, 150), 1)

        try:
            width = int(cv2.getWindowImageRect(window_name)[2])
            height = int(cv2.getWindowImageRect(window_name)[3])
            resized_frame = cv2.resize(frame, (width, height))
            cv2.imshow(window_name, resized_frame)
        except cv2.error:
            break

        key = cv2.waitKey(1)
        if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

if __name__ == "__main__":
    main()
