import mediapipe as mp
import cv2
import os
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from collections import deque
import warnings
warnings.filterwarnings("ignore")
import pyttsx3
import threading

# =========================
# CONFIGURATION
# =========================
DATASET_PATH      = r"C:\Users\USER\PycharmProjects\PythonProject\dataset"
REAL_FRUITS_PATH  = r"C:\Users\USER\PycharmProjects\PythonProject\Real fruit"
CATEGORIES        = ["Apple","Avocado","Banana","Cherry","Coconut","Grapes","Mango","Orange","peach","Pear","Pineapple"]
CONFIDENCE_THRESHOLD = 0.55
SMOOTHING_WINDOW     = 20
OVERLAY_SIZE         = (180, 180)
OVERLAY_PADDING      = 15
FADE_SPEED           = 0.06

# =========================
# TEXT TO SPEECH (FIXED)
# =========================
def speak(text):
    def run():
        engine = pyttsx3.init()
        engine.setProperty("rate", 155)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    threading.Thread(target=run, daemon=True).start()

spoken_fruit = ""
last_spoken_time = 0
SPEAK_DELAY = 2

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands_static = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)
hands_live   = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.5)

# =========================
# LOAD FRUIT IMAGES
# =========================
fruit_images = {}
for category in CATEGORIES:
    for ext in ["jpg","jpeg","png"]:
        path = os.path.join(REAL_FRUITS_PATH, f"{category}.{ext}")
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                fruit_images[category] = cv2.resize(img, OVERLAY_SIZE)
                break

# =========================
# FEATURE EXTRACTION
# =========================
FINGER_TIPS  = [4, 8, 12, 16, 20]
FINGER_BASES = [2, 5,  9, 13, 17]

def get_features(lms_list):
    pts = np.array(lms_list)
    xy  = pts[:, :2]
    wrist  = xy[0]
    coords = xy - wrist
    scale  = np.linalg.norm(coords[9]) + 1e-6
    coords /= scale

    flat = coords.flatten()
    dists = np.linalg.norm(coords, axis=1)

    return np.concatenate([flat, dists])

def extract_from_image(image):
    if image is None: return None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_static.process(rgb)
    if not results.multi_hand_landmarks: return None
    lms = [(lm.x, lm.y) for lm in results.multi_hand_landmarks[0].landmark]
    return get_features(lms)

# =========================
# LOAD DATA
# =========================
X, y = [], []

for label, category in enumerate(CATEGORIES):
    folder = os.path.join(DATASET_PATH, category)
    if not os.path.exists(folder): continue

    for file in os.listdir(folder):
        if file.lower().endswith(('.png','.jpg','.jpeg')):
            feats = extract_from_image(cv2.imread(os.path.join(folder, file)))
            if feats is not None:
                X.append(feats)
                y.append(label)

X, y = np.array(X), np.array(y)

# =========================
# TRAIN MODEL
# =========================
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", probability=True))
])
model.fit(X, y)

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

pred_history = deque(maxlen=SMOOTHING_WINDOW)
last_confirmed = None

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_live.process(rgb)

    confident = False

    if results.multi_hand_landmarks:
        all_probas = []

        for hl in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
            lms = [(lm.x, lm.y) for lm in hl.landmark]
            feats = get_features(lms)
            proba = model.predict_proba(feats.reshape(1,-1))[0]
            all_probas.append(proba)

        proba = max(all_probas, key=lambda p: np.max(p))
        pred = np.argmax(proba)
        conf = np.max(proba)

        pred_history.append(pred)
        final_pred = max(set(pred_history), key=pred_history.count)

        if conf > CONFIDENCE_THRESHOLD:
            last_confirmed = CATEGORIES[final_pred]
            confident = True

    # =========================
    # SPEAK FIX
    # =========================
    current_time = time.time()

    if confident:
        if last_confirmed != spoken_fruit or (current_time - last_spoken_time > SPEAK_DELAY):
            speak(last_confirmed)
            spoken_fruit = last_confirmed
            last_spoken_time = current_time
    else:
        spoken_fruit = ""

    # =========================
    # DISPLAY
    # =========================
    if last_confirmed:
        cv2.putText(frame, last_confirmed, (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Fruit Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
