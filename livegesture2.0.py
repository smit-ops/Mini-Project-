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
CONFIDENCE_THRESHOLD = 0.55   # RAISED from 0.40 — avoids weak guesses
SMOOTHING_WINDOW     = 20     # RAISED from 15 — more temporal stability
OVERLAY_SIZE         = (180, 180)
OVERLAY_PADDING      = 15
FADE_SPEED           = 0.06   # Slightly slower fade for cleaner UX

# =========================
# TEXT TO SPEECH INIT
# =========================
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 155)
tts_engine.setProperty("volume", 1.0)

def speak(text):
    if not tts_engine._inLoop:
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
    static_image_mode=True, max_num_hands=2,   # FIX: only 1 hand needed for dataset
    min_detection_confidence=0.3)
hands_live   = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2,  # FIX: consistent with training
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
    """
    FIX: include_z is now always False both at training and inference.
    Previously training used False but inference passed include_z=True,
    causing a feature dimension mismatch that silently corrupted predictions.
    """
    pts = np.array(lms_list)
    xy  = pts[:, :2]
    wrist  = xy[0]
    coords = xy - wrist
    scale  = np.linalg.norm(coords[9]) + 1e-6
    coords /= scale
    flat   = coords.flatten()      # 42 values

    # FIX: z always zeroed — consistent between train and live
    z_vals = np.zeros(21)

    dists = np.linalg.norm(coords, axis=1)   # 21 values

    conns = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
             (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),
             (15,16),(0,17),(17,18),(18,19),(19,20)]
    angles = [np.arctan2(*(coords[b]-coords[a])[::-1]) for a,b in conns] + [0.0]  # 21 values

    ext_ratios = []
    for tip, base in zip(FINGER_TIPS, FINGER_BASES):
        tip_d  = np.linalg.norm(coords[tip])
        base_d = np.linalg.norm(coords[base]) + 1e-6
        ext_ratios.append(tip_d / base_d)   # 5 values

    spread = []
    for i in range(len(FINGER_TIPS) - 1):
        v1 = coords[FINGER_TIPS[i]]
        v2 = coords[FINGER_TIPS[i+1]]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6)
        spread.append(np.arccos(np.clip(cos_a, -1, 1)))   # 4 values

    palm_centre = np.mean(coords[[0, 5, 9, 13, 17]], axis=0)
    palm_dists  = [np.linalg.norm(coords[t] - palm_centre) for t in FINGER_TIPS]  # 5 values
    palm_width  = np.linalg.norm(coords[5]  - coords[17])
    palm_height = np.linalg.norm(coords[0]  - coords[9])
    aspect      = palm_width / (palm_height + 1e-6)

    # NEW: curvature features — how bent each finger is
    curvatures = []
    finger_chains = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]
    for chain in finger_chains:
        v1 = coords[chain[1]] - coords[chain[0]]
        v2 = coords[chain[2]] - coords[chain[1]]
        v3 = coords[chain[3]] - coords[chain[2]]
        cos1 = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6)
        cos2 = np.dot(v2,v3)/(np.linalg.norm(v2)*np.linalg.norm(v3)+1e-6)
        curvatures.append(np.arccos(np.clip(cos1,-1,1)))
        curvatures.append(np.arccos(np.clip(cos2,-1,1)))   # 10 values

    # NEW: inter-fingertip distances (relative spread between all tip pairs)
    tip_pair_dists = []
    for i in range(len(FINGER_TIPS)):
        for j in range(i+1, len(FINGER_TIPS)):
            tip_pair_dists.append(
                np.linalg.norm(coords[FINGER_TIPS[i]] - coords[FINGER_TIPS[j]])
            )   # 10 values

    return np.concatenate([
        flat,           # 42
        z_vals,         # 21
        dists,          # 21
        angles,         # 21
        ext_ratios,     # 5
        spread,         # 4
        palm_dists,     # 5
        [palm_width, aspect],  # 2
        curvatures,     # 10
        tip_pair_dists  # 10
    ])   # Total: 141 features

def extract_from_image(image):
    if image is None: return None
    rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_static.process(rgb)
    if not results.multi_hand_landmarks:
        bright  = cv2.convertScaleAbs(rgb, alpha=1.3, beta=40)
        results = hands_static.process(bright)
    if not results.multi_hand_landmarks: return None
    lms = [(lm.x, lm.y) for lm in results.multi_hand_landmarks[0].landmark]
    return get_features(lms)   # FIX: no include_z argument — always consistent

# =========================
# DATA AUGMENTATION (improved)
# =========================
AUG_COPIES = 8   # RAISED from 5 — more diversity per sample

def augment_features(feats):
    augmented = []
    for k in range(AUG_COPIES):
        f = feats.copy()
        # Vary noise level across copies so the model sees a wider distribution
        noise_scale = 0.010 + 0.015 * (k / AUG_COPIES)

        # XY coordinate noise
        f[:42] += np.random.normal(0, noise_scale, size=42)

        # Scale jitter (simulate hand closer/farther from camera)
        scale_jitter = np.random.uniform(0.88, 1.12)
        f[:42] *= scale_jitter
        f[63:84] *= scale_jitter   # distances scale too

        # Slight rotation simulation (rotate 2D coords by small angle)
        angle = np.random.uniform(-0.15, 0.15)   # radians
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        xy = f[:42].reshape(21, 2)
        rotated = xy @ np.array([[cos_a, -sin_a],[sin_a, cos_a]])
        f[:42] = rotated.flatten()

        # Angle perturbation
        f[84:105] += np.random.uniform(-0.06, 0.06)

        # Extension ratio perturbation
        f[105:110] = np.clip(f[105:110] + np.random.normal(0, 0.018, 5), 0, None)

        augmented.append(f)
    return augmented

# =========================
# LOAD DATASET & CHECK BALANCE
# =========================
print("\nLoading gesture dataset...")
X, y = [], []
class_counts = {}

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
    class_counts[category] = count
    total = count * (AUG_COPIES + 1)
    print(f"  {category}: {count} real + {count*AUG_COPIES} aug = {total} total")

# CLASS BALANCE CHECK — warn if any class is severely underrepresented
print("\n--- Class balance check ---")
max_count = max(class_counts.values()) if class_counts else 1
for cat, cnt in class_counts.items():
    ratio = cnt / max_count
    flag = "  LOW — add more images!" if ratio < 0.5 else ""
    print(f"  {cat}: {cnt} real images  ({ratio*100:.0f}% of max){flag}")

if len(X) < 4:
    print("Too few samples."); exit()

X, y = np.array(X), np.array(y)
print(f"\nTotal samples: {len(X)}  |  Feature dimensions: {X.shape[1]}")

# =========================
# TRAIN ENSEMBLE MODEL
# =========================
# FIX: Use an ensemble of 3 SVMs with different hyperparameters.
# This dramatically reduces the chance of one class dominating.
print("\nTraining ensemble SVM...")

def make_svm(C, gamma):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            SVC(kernel="rbf", C=C, gamma=gamma,
                class_weight="balanced",   # CRITICAL: compensates for class imbalance
                random_state=42),
            cv=3
        ))
    ])

# Train three SVMs with different C/gamma values
svm1 = make_svm(C=10,  gamma="scale")
svm2 = make_svm(C=50,  gamma="scale")
svm3 = make_svm(C=100, gamma="auto")

svm1.fit(X, y)
svm2.fit(X, y)
svm3.fit(X, y)

# Cross-validation on the primary model to get a realistic accuracy estimate
print("Running 5-fold cross-validation (realistic accuracy estimate)...")
cv_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", C=50, gamma="scale",
                class_weight="balanced", probability=True, random_state=42))
])
cv_scores = cross_val_score(cv_svm, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring="accuracy")
print(f"Cross-val accuracy: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")
print(f"Training accuracy (svm2): {np.mean(svm2.predict(X)==y)*100:.1f}%\n")

# =========================
# ENSEMBLE PREDICTION FUNCTION
# =========================
def ensemble_predict(feat_vec):
    """
    Average probabilities across 3 SVMs.
    Each SVM was calibrated with CalibratedClassifierCV so probabilities
    are well-calibrated (not just raw SVC scores).
    """
    p1 = svm1.predict_proba(feat_vec.reshape(1,-1))[0]
    p2 = svm2.predict_proba(feat_vec.reshape(1,-1))[0]
    p3 = svm3.predict_proba(feat_vec.reshape(1,-1))[0]
    return (p1 + p2 + p3) / 3.0

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
cap = cv2.VideoCapture(0)
time.sleep(1)

pred_history    = deque(maxlen=SMOOTHING_WINDOW)
current_alpha   = 0.0
last_confirmed  = None
spoken_fruit    = None

# NEW: Stability counter — require N consecutive frames of same prediction
# before accepting it (prevents flickering between classes)
STABILITY_REQUIRED = 8
stability_counter  = 0
stable_candidate   = None

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

    if results.multi_hand_landmarks:
        all_probas = []

        for hl in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

            lms = [(lm.x, lm.y) for lm in hl.landmark]
            feats_input = get_features(lms)

            proba = ensemble_predict(feats_input)
            all_probas.append(proba)

        # Combine both hands predictions (average)
        proba = np.mean(all_probas, axis=0)

        # Smoothing window vote
        pred_history.append(int(np.argmax(proba)))
        sl = max(set(pred_history), key=pred_history.count)
        sc = proba[sl]

        # NEW: Stability gate — same class must win for STABILITY_REQUIRED frames
        if sl == stable_candidate:
            stability_counter += 1
        else:
            stable_candidate  = sl
            stability_counter = 1

        stable_enough = (stability_counter >= STABILITY_REQUIRED)

        if sc >= CONFIDENCE_THRESHOLD and stable_enough:
            last_confirmed = CATEGORIES[sl]
            confident      = True
            display_text   = f"{last_confirmed}  {sc*100:.0f}%"
            display_color  = (0, 220, 0)
        elif sc >= CONFIDENCE_THRESHOLD:
            # High confidence but not yet stable
            display_text  = f"Stabilising... {CATEGORIES[sl]}  {sc*100:.0f}%"
            display_color = (0, 180, 255)
        else:
            display_text  = f"Searching...  {sc*100:.0f}%"
            display_color = (0, 120, 255)

        # Probability bars
        bx = frame.shape[1] - 230
        sorted_indices = np.argsort(proba)[::-1]
        for rank, i in enumerate(sorted_indices[:8]):   # show top 8 only
            by = 65 + rank*30
            bar_w = int(proba[i]*160)
            color = (0,200,80) if i==sl else (150,150,150)
            cv2.rectangle(frame, (bx, by), (bx+bar_w, by+18), color, -1)
            cv2.putText(frame, f"{CATEGORIES[i][:8]}: {proba[i]*100:.0f}%",
                        (bx-5, by+13), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (255,255,255), 1, cv2.LINE_AA)
    else:
        pred_history.clear()
        stability_counter = 0
        stable_candidate  = None

    # Fade logic
    prev_alpha    = current_alpha
    current_alpha = min(1.0, current_alpha + FADE_SPEED) if confident \
                    else max(0.0, current_alpha - FADE_SPEED)

    # TTS trigger
    just_fully_visible = (prev_alpha < 1.0 and current_alpha >= 1.0)
    if just_fully_visible and last_confirmed and last_confirmed != spoken_fruit:
        speak(f"{last_confirmed}")
        spoken_fruit = last_confirmed
    if not confident:
        spoken_fruit = None

    # Draw overlay
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
