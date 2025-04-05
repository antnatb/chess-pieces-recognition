import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from joblib import dump

DATA_DIR = "data/detected"
IMAGE_SIZE = (32, 64)  # Adjusted for 2:1 height-to-width ratio  # Standardized size for all pieces

def extract_features(image):
    # Resize image
    image = cv2.resize(image, IMAGE_SIZE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. HOG features
    hog_features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
                       block_norm='L2-Hys', visualize=False, feature_vector=True)

    # 2. Hu Moments
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)  # log scale

    # Combine all features into one vector
    return np.hstack((hog_features, hu_moments))

def load_data():
    X = []
    y = []
    for color in ["black", "white"]:
        color_path = os.path.join(DATA_DIR, color)
        if not os.path.isdir(color_path):
            continue
        for label in sorted(os.listdir(color_path)):
            label_path = os.path.join(color_path, label)
            if not os.path.isdir(label_path):
                continue
            for file in os.listdir(label_path):
                if file.endswith(".png") or file.endswith(".jpg"):
                    path = os.path.join(label_path, file)
                    img = cv2.imread(path)
                    if img is None:
                        continue
                    features = extract_features(img)
                    X.append(features)
                    y.append(f"{color}_{label}")
    return np.array(X), np.array(y)

def main():
    print("Loading data and extracting features...")
    X, y = load_data()
    print(f"Dataset size: {len(X)} samples")
    if len(X) == 0:
        print("No data found. Please check your dataset path and labels.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("Training SVM classifier...")
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    dump(clf, "svm_chess_classifier.joblib")
    print("Model saved as 'svm_chess_classifier.joblib'")

if __name__ == "__main__":
    main()
