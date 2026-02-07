import mediapipe as mp
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog

# =========================
# CONFIG
# =========================
DATASET_PATH = r"C:\Users\USER\PycharmProjects\PythonProject\dataset"
CATEGORIES = ["apple", "banana"]
IMG_SIZE = 64
THRESHOLD = 0.85   # confidence threshold

# =========================
# LOAD DATASET
# =========================
data = []
labels = []

for label, category in enumerate(CATEGORIES):

    folder_path = os.path.join(DATASET_PATH, category)

    for file in os.listdir(folder_path):

        img_path = os.path.join(folder_path, file)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        features = hog(
            img,
            orientations=9,
            pixels_per_cell=(8,8),
            cells_per_block=(2,2),
            block_norm='L2-Hys'
        )

        data.append(features)
        labels.append(label)

print("✅ Images loaded:", len(data))

X = np.array(data)
y = np.array(labels)

# =========================
# TRAIN MODEL
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

print("✅ Model trained successfully")

y_pred = model.predict(X_test)
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))

# =========================
# MEDIAPIPE HAND SETUP
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# =========================
# CAMERA START
# =========================
print("\n🎥 Starting camera... Press Q to exit")

cap = cv2.VideoCapture(0)

last_predictions = []

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera not working")
        break

    label_text = "No Hand"

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape

            x_list = []
            y_list = []

            for lm in hand_landmarks.landmark:
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)

            if x_max > x_min and y_max > y_min:

                hand_img = frame[y_min:y_max, x_min:x_max]

                if hand_img is not None and hand_img.size != 0:

                    gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)

                    # VERY IMPORTANT resize (fixes previous error)
                    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
                    gray = gray / 255.0

                    features = hog(
                        gray,
                        orientations=9,
                        pixels_per_cell=(8,8),
                        cells_per_block=(2,2),
                        block_norm='L2-Hys'
                    )

                    features = features.reshape(1, -1)

                    probs = model.predict_proba(features)[0]
                    max_conf = max(probs)
                    pred = model.predict(features)[0]

                    # ===== ULTRA STABLE DETECTION =====
                    if max_conf < THRESHOLD:
                        current_prediction = "Unknown"
                    elif pred == 0:
                        current_prediction = "Apple"
                    else:
                        current_prediction = "Banana"

                    last_predictions.append(current_prediction)

                    if len(last_predictions) > 5:
                        last_predictions.pop(0)

                    label_text = max(set(last_predictions), key=last_predictions.count)

    cv2.putText(frame, label_text, (50,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.imshow("Fruit Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
