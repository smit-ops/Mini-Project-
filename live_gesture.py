import mediapipe as mp
import cv2
import os
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import deque
import warnings
warnings.filterwarnings("ignore")

# =========================
# CONFIGURATION
# =========================
DATASET_PATH      = r"C:\Users\Smit\PyCharmMiscProject\dataset"
REAL_FRUITS_PATH  = r"C:\Users\Smit\PyCharmMiscProject\real_fruits"   # ← NEW folder
CATEGORIES        = ["apple", "grapes", "mango", "pineapple","pear","banana","cherry","avocado","coconut"]
CONFIDENCE_THRESHOLD = 0.65   # overlay + text only appear above this threshold
SMOOTHING_WINDOW     = 15          # more frames = smoother, more stable result

OVERLAY_SIZE    = (180, 180)   # px: width x height of the fruit image corner box
OVERLAY_PADDING = 15           # gap from screen edge in px
FADE_SPEED      = 0.08         # fade-in speed per frame (0.01=slow  0.2=fast)

# =========================
# MEDIAPIPE INIT
# =========================
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands_static = mp_hands.Hands(
    static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

hands_live = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2,          # ← detect both hands
    min_detection_confidence=0.6, min_tracking_confidence=0.5)

# =========================
# LOAD REAL FRUIT IMAGES
# =========================
# Create a folder:  C:\Users\Smit\PyCharmMiscProject\real_fruits\
# Place ONE image per fruit named exactly:
#   apple.jpg   grapes.jpg   mango.jpg   pineapple.jpg
# (.jpeg .png .bmp .webp also work)
fruit_images = {}
print("Fruit images loading...")
for category in CATEGORIES:
    for ext in ["jpg", "jpeg", "png", "bmp", "webp"]:
        path = os.path.join(REAL_FRUITS_PATH, f"{category}.{ext}")
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                fruit_images[category] = cv2.resize(img, OVERLAY_SIZE)
                print(f"  OK  {category}.{ext}")
                break
    else:
        print(f"  MISSING  {category}  -> add {category}.jpg to real_fruits/")

# =========================
# FEATURE EXTRACTION (84-dim)
# =========================
def get_features(lms_list):
    wrist  = np.array(lms_list[0])
    coords = np.array(lms_list) - wrist
    scale  = np.linalg.norm(coords[9]) + 1e-6
    coords /= scale
    flat   = coords.flatten()
    dists  = np.linalg.norm(coords, axis=1)
    conns  = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
              (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),
              (15,16),(0,17),(17,18),(18,19),(19,20)]
    angles = [np.arctan2(*(coords[b]-coords[a])[::-1]) for a,b in conns] + [0.0]
    return np.concatenate([flat, dists, angles])

def extract_from_image(image):
    if image is None: return None
    rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_static.process(rgb)
    if not results.multi_hand_landmarks:
        bright  = cv2.convertScaleAbs(rgb, alpha=1.3, beta=40)
        results = hands_static.process(bright)
    if not results.multi_hand_landmarks: return None
    return get_features([(lm.x, lm.y) for lm in results.multi_hand_landmarks[0].landmark])

# =========================
# LOAD DATASET & TRAIN SVM
# =========================
print("\nLoading gesture dataset...")
X, y = [], []
for label, category in enumerate(CATEGORIES):
    folder = os.path.join(DATASET_PATH, category)
    if not os.path.exists(folder):
        print(f"  MISSING folder: {folder}"); continue
    files = [f for f in os.listdir(folder)
             if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]
    count = 0
    for file in files:
        feats = extract_from_image(cv2.imread(os.path.join(folder, file)))
        if feats is not None:
            X.append(feats); y.append(label); count += 1
    print(f"  {category}: {count}/{len(files)} hands")

if len(X) < 4:
    print("Too few samples — ensure dataset images show hands."); exit()

X, y = np.array(X), np.array(y)
model = Pipeline([("scaler", StandardScaler()),
                  ("clf", SVC(kernel="rbf", C=50, gamma="scale",
                              class_weight="balanced",   # handles unequal class sizes
                              probability=True, random_state=42))])
model.fit(X, y)
print(f"\nSVM trained  accuracy: {np.mean(model.predict(X)==y)*100:.1f}%\n")

# =========================
# OVERLAY HELPER
# =========================
def draw_fruit_overlay(frame, fruit_img, alpha):
    fh, fw  = frame.shape[:2]
    oh, ow  = OVERLAY_SIZE[1], OVERLAY_SIZE[0]
    x1 = fh - oh - OVERLAY_PADDING
    y1 = fw - ow - OVERLAY_PADDING
    x2, y2 = x1 + oh, y1 + ow
    if x1 < 0 or y1 < 0: return frame
    pad = 8
    # card background
    cv2.rectangle(frame, (y1-pad, x1-pad), (y2+pad, x2+pad), (20,20,20), -1)
    cv2.rectangle(frame, (y1-pad, x1-pad), (y2+pad, x2+pad), (70,70,70),  1)
    # blend image
    roi = frame[x1:x2, y1:y2]
    frame[x1:x2, y1:y2] = cv2.addWeighted(fruit_img, alpha, roi, 1-alpha, 0)
    return frame

# =========================
# LIVE CAMERA LOOP
# =========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(1)

pred_history   = deque(maxlen=SMOOTHING_WINDOW)
current_alpha  = 0.0
last_confirmed = None

print("Camera running — press Q to quit\n")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame   = cv2.flip(frame, 1)
    results = hands_live.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    display_text  = "No hand detected"
    display_color = (120, 120, 120)
    confident     = False

    if results.multi_hand_landmarks:
        # ── process ALL detected hands (up to 2) and average probabilities ──
        all_probas = []
        for hl in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
            lms = [(lm.x, lm.y) for lm in hl.landmark]
            all_probas.append(model.predict_proba(get_features(lms).reshape(1,-1))[0])

        # average across both hands for a more stable prediction
        proba = np.mean(all_probas, axis=0)

        pred_history.append(int(np.argmax(proba)))
        sl   = max(set(pred_history), key=pred_history.count)
        sc   = proba[sl]

        if sc >= CONFIDENCE_THRESHOLD:          # only triggers above 65%
            last_confirmed = CATEGORIES[sl]
            confident      = True
            display_text   = f"{last_confirmed}  {sc*100:.0f}%"
            display_color  = (0, 220, 0)
        else:
            display_text  = f"Searching...  {sc*100:.0f}%"
            display_color = (0, 120, 255)

        # probability bars
        bx = frame.shape[1] - 220
        for i, (cat, p) in enumerate(zip(CATEGORIES, proba)):
            by = 65 + i*38
            cv2.rectangle(frame, (bx, by),
                          (bx + int(p*150), by+22),
                          (0,200,80) if i==sl else (170,170,170), -1)
            cv2.putText(frame, f"{cat[:7]}: {p*100:.0f}%",
                        (bx-5, by+16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.44, (255,255,255), 1, cv2.LINE_AA)
    else:
        pred_history.clear()

    # ── fade in / out ──────────────────────────────────────────────────────
    current_alpha = min(1.0, current_alpha + FADE_SPEED) if confident \
                    else max(0.0, current_alpha - FADE_SPEED)

    # ── fruit image overlay ────────────────────────────────────────────────
    if last_confirmed and last_confirmed in fruit_images and current_alpha > 0:
        frame = draw_fruit_overlay(frame, fruit_images[last_confirmed], current_alpha)

        # name badge — bottom-LEFT corner (opposite side from the fruit photo)
        fh, fw  = frame.shape[:2]
        text    = last_confirmed.upper()
        font    = cv2.FONT_HERSHEY_DUPLEX
        scale   = 1.2
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        lx = OVERLAY_PADDING                  # left edge
        ly = fh - OVERLAY_PADDING             # bottom edge
        # dark pill background so text is always readable over any camera feed
        cv2.rectangle(frame,
                      (lx - 8,      ly - th - 10),
                      (lx + tw + 8, ly + baseline),
                      (20, 20, 20), -1)
        cv2.rectangle(frame,
                      (lx - 8,      ly - th - 10),
                      (lx + tw + 8, ly + baseline),
                      (70, 70, 70), 1)
        tmp = frame.copy()
        cv2.putText(tmp, text, (lx, ly),
                    font, scale, (255, 215, 50), thickness, cv2.LINE_AA)
        frame = cv2.addWeighted(tmp, current_alpha, frame, 1-current_alpha*0.4, 0)

    # ── top label bar ──────────────────────────────────────────────────────
    cv2.rectangle(frame, (0,0), (frame.shape[1], 55), (25,25,25), -1)
    cv2.putText(frame, display_text, (15, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, display_color, 2, cv2.LINE_AA)

    cv2.imshow("Fruit Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()