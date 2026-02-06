import cv2
import mediapipe as mp
import numpy as np
import joblib
from skimage.feature import hog

IMG_SIZE = 64
THRESHOLD = 0.8

# Load trained model
model = joblib.load("gesture_model.pkl")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    label_text = "Show Gesture"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_list, y_list = [], []

            for lm in hand_landmarks.landmark:
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)

            hand = frame[y_min:y_max, x_min:x_max]

            if hand.size != 0:
                gray = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
                gray = gray / 255.0

                features = hog(
                    gray,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    block_norm='L2-Hys'
                )

                features = features.reshape(1, -1)
                probs = model.predict_proba(features)[0]
                confidence = max(probs)
                pred = model.predict(features)[0]

                if confidence > THRESHOLD:
                    label_text = "Apple" if pred == 0 else "Banana"
                else:
                    label_text = "Unknown"

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)

    cv2.putText(
        frame, label_text, (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2
    )

    cv2.imshow("Live Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
