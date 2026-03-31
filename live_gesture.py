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
import pyttsx3
import threading

# =========================
# CONFIGURATION
# =========================
DATASET_PATH      = r"C:\Users\Smit\PyCharmMiscProject\dataset"
REAL_FRUITS_PATH  = r"C:\Users\Smit\PyCharmMiscProject\real_fruits"
CATEGORIES        = ["apple","grapes","mango","pineapple","pear","cherry","coconut","avocado","peach","orange"]
CONFIDENCE_THRESHOLD = 0.40
SMOOTHING_WINDOW     = 15
OVERLAY_SIZE         = (180, 180)
OVERLAY_PADDING      = 15
FADE_SPEED           = 0.08

# =========================
# TEXT TO SPEECH INIT
# =========================
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 155)
tts_engine.setProperty("volume", 1.0)

def speak(text):
    """Run TTS in a background thread so camera never freezes."""
    def _run():
        tts_engine.say(text)
        tts_engine.runAndWait()
    threading.Thread(target=_run, daemon=True).start()

# =========================
# MEDIAPIPE INIT
# =========================
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands_static = mp_hands.Hands(
    static_image_mode=True,  max_num_hands=2, min_detection_confidence=0.3)
hands_live   = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2,
    min_detection_confidence=0.6, min_tracking_confidence=0.5)

# =========================
# LOAD REAL FRUIT IMAGES
# =========================
fruit_images = {}
print("Loading fruit images...")
for category in CATEGORIES:
    for ext in ["jpg","jpeg","png","bmp","webp"]:
        path = os.path.join(REAL_FRUITS_PATH, f"{category}.{ext}")
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                fruit_images[category] = cv2.resize(img, OVERLAY_SIZE)
                print(f"  OK  {category}.{ext}")
                break
    else:
        print(f"  MISSING  {category}")

# =========================
# FEATURE EXTRACTION
# =========================
FINGER_TIPS  = [4, 8, 12, 16, 20]
FINGER_BASES = [2, 5,  9, 13, 17]
FINGER_MIDS  = [3, 7, 11, 15, 19]

def get_features(lms_list, include_z=False):
    pts = np.array(lms_list)
    xy  = pts[:, :2]
    wrist  = xy[0]
    coords = xy - wrist
    scale  = np.linalg.norm(coords[9]) + 1e-6
    coords /= scale
    flat   = coords.flatten()
    if include_z and pts.shape[1] == 3:
        z_vals = pts[:, 2] - pts[0, 2]
        z_vals = z_vals / (np.std(z_vals) + 1e-6)
    else:
        z_vals = np.zeros(21)
    dists = np.linalg.norm(coords, axis=1)
    conns = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
             (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),
             (15,16),(0,17),(17,18),(18,19),(19,20)]
    angles = [np.arctan2(*(coords[b]-coords[a])[::-1]) for a,b in conns] + [0.0]
    ext_ratios = []
    for tip, base in zip(FINGER_TIPS, FINGER_BASES):
        tip_d  = np.linalg.norm(coords[tip])
        base_d = np.linalg.norm(coords[base]) + 1e-6
        ext_ratios.append(tip_d / base_d)
    spread = []
    for i in range(len(FINGER_TIPS) - 1):
        v1 = coords[FINGER_TIPS[i]]
        v2 = coords[FINGER_TIPS[i+1]]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6)
        spread.append(np.arccos(np.clip(cos_a, -1, 1)))
    palm_centre = np.mean(coords[[0, 5, 9, 13, 17]], axis=0)
    palm_dists  = [np.linalg.norm(coords[t] - palm_centre) for t in FINGER_TIPS]
    palm_width  = np.linalg.norm(coords[5]  - coords[17])
    palm_height = np.linalg.norm(coords[0]  - coords[9])
    aspect      = palm_width / (palm_height + 1e-6)
    return np.concatenate([flat, z_vals, dists, angles, ext_ratios, spread, palm_dists, [palm_width, aspect]])

def extract_from_image(image):
    if image is None: return None
    rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_static.process(rgb)
    if not results.multi_hand_landmarks:
        bright  = cv2.convertScaleAbs(rgb, alpha=1.3, beta=40)
        results = hands_static.process(bright)
    if not results.multi_hand_landmarks: return None
    lms = [(lm.x, lm.y) for lm in results.multi_hand_landmarks[0].landmark]
    return get_features(lms, include_z=False)

# =========================
# DATA AUGMENTATION
# =========================
AUG_COPIES = 5

def augment_features(feats):
    augmented = []
    for _ in range(AUG_COPIES):
        f = feats.copy()
        f[:42]    += np.random.normal(0, 0.015, size=42)
        f[63:84]  *= np.random.uniform(0.92, 1.08)
        f[84:105] += np.random.uniform(-0.08, 0.08)
        f[105:110] = np.clip(f[105:110] + np.random.normal(0, 0.02, 5), 0, None)
        augmented.append(f)
    return augmented

# =========================
# LOAD DATASET & TRAIN
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
            X.append(feats); y.append(label)
            for aug in augment_features(feats):
                X.append(aug); y.append(label)
            count += 1
    print(f"  {category}: {count} real  +  {count*AUG_COPIES} augmented  =  {count*(AUG_COPIES+1)} total")

if len(X) < 4:
    print("Too few samples."); exit()

X, y = np.array(X), np.array(y)
model = Pipeline([("scaler", StandardScaler()),
                  ("clf", SVC(kernel="rbf", C=50, gamma="scale",
                              class_weight="balanced",
                              probability=True, random_state=42))])
model.fit(X, y)
print(f"\nSVM trained  —  training accuracy: {np.mean(model.predict(X)==y)*100:.1f}%\n")

# =========================
# OVERLAY HELPER
# =========================
def draw_fruit_overlay(frame, fruit_img, alpha):
    fh, fw = frame.shape[:2]
    oh, ow = OVERLAY_SIZE[1], OVERLAY_SIZE[0]
    x1, y1 = fh - oh - OVERLAY_PADDING, fw - ow - OVERLAY_PADDING
    x2, y2 = x1 + oh, y1 + ow
    if x1 < 0 or y1 < 0: return frame
    pad = 8
    cv2.rectangle(frame, (y1-pad, x1-pad), (y2+pad, x2+pad), (20,20,20), -1)
    cv2.rectangle(frame, (y1-pad, x1-pad), (y2+pad, x2+pad), (70,70,70),  1)
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
last_confirmed = None   # what model sees (prediction only)
spoken_fruit   = None   # what was last spoken (voice only)

print("Camera running — press Q to quit\n")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame   = cv2.flip(frame, 1)
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_live.process(rgb)

    display_text  = "No hand detected"
    display_color = (120, 120, 120)
    confident     = False

    # ── PREDICTION (unchanged from working version) ───────────────────────
    if results.multi_hand_landmarks:
        all_probas = []
        for hl in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
            lms         = [(lm.x, lm.y, lm.z) for lm in hl.landmark]
            feats_input = get_features(lms, include_z=True)[:X.shape[1]]
            all_probas.append(model.predict_proba(feats_input.reshape(1,-1))[0])

        proba = np.mean(all_probas, axis=0)
        pred_history.append(int(np.argmax(proba)))
        sl = max(set(pred_history), key=pred_history.count)
        sc = proba[sl]

        if sc >= CONFIDENCE_THRESHOLD:
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
            by = 65 + i*34
            cv2.rectangle(frame, (bx, by), (bx+int(p*150), by+20),
                          (0,200,80) if i==sl else (170,170,170), -1)
            cv2.putText(frame, f"{cat[:7]}: {p*100:.0f}%",
                        (bx-5, by+14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (255,255,255), 1, cv2.LINE_AA)
    else:
        pred_history.clear()

    # ── FADE (unchanged from working version) ─────────────────────────────
    prev_alpha    = current_alpha
    current_alpha = min(1.0, current_alpha + FADE_SPEED) if confident \
                    else max(0.0, current_alpha - FADE_SPEED)

    # ── VOICE (added on top — never touches prediction or display vars) ───
    # Speaks once when image becomes fully visible
    # spoken_fruit resets when confidence drops so same fruit speaks again
    just_fully_visible = (prev_alpha < 1.0 and current_alpha >= 1.0)
    if just_fully_visible and last_confirmed and last_confirmed != spoken_fruit:
        speak(f"{last_confirmed}")
        spoken_fruit = last_confirmed
    if not confident:
        spoken_fruit = None   # reset so it speaks again next time hand shows

    # ── DRAW (unchanged from working version) ─────────────────────────────
    if last_confirmed and last_confirmed in fruit_images and current_alpha > 0:
        frame = draw_fruit_overlay(frame, fruit_images[last_confirmed], current_alpha)
        fh, fw = frame.shape[:2]
        text   = last_confirmed.upper()
        font, scale, thick = cv2.FONT_HERSHEY_DUPLEX, 1.2, 2
        (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
        lx, ly = OVERLAY_PADDING, fh - OVERLAY_PADDING
        cv2.rectangle(frame, (lx-8, ly-th-10), (lx+tw+8, ly+bl), (20,20,20), -1)
        cv2.rectangle(frame, (lx-8, ly-th-10), (lx+tw+8, ly+bl), (70,70,70),   1)
        tmp = frame.copy()
        cv2.putText(tmp, text, (lx, ly), font, scale, (255,215,50), thick, cv2.LINE_AA)
        frame = cv2.addWeighted(tmp, current_alpha, frame, 1-current_alpha*0.4, 0)

    cv2.rectangle(frame, (0,0), (frame.shape[1], 55), (25,25,25), -1)
    cv2.putText(frame, display_text, (15,38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, display_color, 2, cv2.LINE_AA)

    cv2.imshow("Fruit Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
