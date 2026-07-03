import cv2
import mediapipe as mp
import math
import numpy as np  # Added for blank frame generation
import mediapipe.python.solutions.hands as mp_hands_module

mp_hands = mp_hands_module
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
# Ensure the path is correct for your system
cap_cat = cv2.VideoCapture(r"C:\Users\admin\Downloads\YTDown_Shorts_scuba-cat-kicau-mania-meme-memes-shorts-_Media_TBxXCyeN09E_001_1080p.mp4")

def points_close(p1, p2, threshold=0.05):
    return math.hypot(p1.x - p2.x, p1.y - p2.y) < threshold

def verify_fist(landmarks):
    wrist = landmarks[0]
    tips = [8, 12, 16, 20]
    return all(points_close(landmarks[t], wrist, 0.15) for t in tips)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    hay_fist = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if verify_fist(hand_landmarks.landmark):
                hay_fist = True

    # Prepare a blank frame to serve as the "off" state
    blank_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

    if hay_fist and cap_cat.isOpened():
        ret_video, frame_cat = cap_cat.read()
        if not ret_video:
            cap_cat.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_video, frame_cat = cap_cat.read()
        
        if ret_video:
            frame_cat = cv2.resize(frame_cat, (frame.shape[1], frame.shape[0]))
            cv2.imshow("SCUBAAA SCUBAAA", frame_cat)
    else:
        # Instantly show black frame when not in a fist
        cv2.imshow("SCUBAAA SCUBAAA", blank_frame)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cap_cat.release()
cv2.destroyAllWindows()