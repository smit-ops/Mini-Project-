import os
import cv2

dataset_path = "dataset/apple"

print("Images found:")

for img_name in os.listdir(dataset_path):
    print(img_name)

    img_path = os.path.join(dataset_path, img_name)
    img = cv2.imread(img_path, 0)

    if img is None:
        print("❌ Could not read image")
    else:
        print("✅ Image loaded successfully")

import cv2
import os
import numpy as np

data = []     # to store image data
labels = []   # to store label (apple)

img_size = 64  # fixed size

apple_dir = r"C:\Users\USER\PycharmProjects\PythonProject\dataset\apple"

for img_name in os.listdir(apple_dir):
    img_path = os.path.join(apple_dir, img_name)


    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Skipped:", img_name)
        continue

    img = cv2.resize(img, (img_size, img_size))

    data.append(img.flatten())
    labels.append(0)
print("Testing single image...")

test_img = cv2.imread(r"C:\Users\USER\OneDrive\Pictures\Screenshots\Screenshot (50).png")
print(test_img)
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# convert to numpy arrays
X = np.array(data)
y = np.array(labels)
# TEMPORARY FIX: create fake second class
X = np.vstack([X, X])
y = np.hstack([y, np.ones(len(y))])

# split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# create SVM model
model = SVC(kernel='linear')

# train the model
model.fit(X_train, y_train)

print("Model trained successfully")
# test prediction using first loaded image
test_img = X_test[0].reshape(1, -1)

prediction = model.predict(test_img)
print("Prediction:", prediction)
