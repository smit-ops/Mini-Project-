import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import joblib

# =========================
# CONFIG
# =========================
DATASET_PATH = "dataset"
CATEGORIES = ["apple", "banana"]
IMG_SIZE = 64

data = []
labels = []

# =========================
# LOAD DATA
# =========================
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
# TRAIN TEST SPLIT
# =========================
X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# TRAIN MODEL
# =========================
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# =========================
# EVALUATE
# =========================
y_pred = model.predict(X_test)
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "gesture_model.pkl")
print("💾 Model saved as gesture_model.pkl")
