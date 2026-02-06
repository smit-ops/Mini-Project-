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
DATASET_PATH = r"C:\Users\Smit\PyCharmMiscProject\dataset"
CATEGORIES = ["apple", "banana"]
IMG_SIZE = 64

# =========================
# LOAD DATA
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
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )
        data.append(features)

        labels.append(label)

print("✅ Images loaded:", len(data))

# =========================
# CONVERT TO NUMPY
# =========================
X = np.array(data)
y = np.array(labels)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)


# =========================
# TRAIN MODEL
# =========================
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

print("✅ Model trained successfully")

# =========================
# TEST ACCURACY
# =========================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("🎯 Accuracy:", accuracy)

# =========================
# TEST ON SINGLE IMAGE
# =========================
img = cv2.imread(r"C:\Users\Smit\Downloads\parth.jpeg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("❌ Image not found")
else:
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    features = features.reshape(1, -1)

    probs = model.predict_proba(features)[0]

    print("Probabilities:", probs)  # 👈 ADD THIS LINE

    max_conf = max(probs)
    pred = model.predict(features)[0]

    THRESHOLD = 0.8



    if max_conf < THRESHOLD:
        print("🤷 Unknown image")
    elif pred == 0:
        print("🍎 Apple")
    else:
        print("🍌 Banana")