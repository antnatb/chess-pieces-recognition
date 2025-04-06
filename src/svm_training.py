import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
from skimage.feature import hog
from joblib import dump

DATA_DIR = "data/detected"
IMAGE_SIZE = (32, 64)  # Adjusted for 2:1 height-to-width ratio

def extract_features(image):
    image = cv2.resize(image, IMAGE_SIZE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. HOG features
    hog_features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                       block_norm='L2-Hys', visualize=False, feature_vector=True)

    # 2. Hu Moments
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)  # log scale

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

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save the best model and transformer")
    args = parser.parse_args()

    print("Loading data and extracting features...")
    X, y = load_data()
    print(f"Dataset size: {len(X)} samples")
    if len(X) == 0:
        print("No data found. Please check your dataset path and labels.")
        return

    k = 5
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    accuracies = []
    best_acc = -1
    best_model = None
    best_lda = None

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}/{k}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        lda = LDA()
        X_train_lda = lda.fit_transform(X_train, y_train)
        X_test_lda = lda.transform(X_test)
        print(f"LDA reduced to {X_train_lda.shape[1]} features")

        clf = SVC(kernel='linear', probability=True)
        clf.fit(X_train_lda, y_train)

        y_pred = clf.predict(X_test_lda)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        if acc > best_acc:
            best_acc = acc
            best_model = clf
            best_lda = lda

        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    print(f"\nAverage Accuracy over {k} folds: {np.mean(accuracies):.4f}")

    if args.save and best_model is not None:
            dump(best_model, "SVC_LDA.joblib")
            dump(best_lda, "lda_transformer.joblib")
            print(f"Model saved as 'SVC_LDA.joblib' with accuracy: {best_acc:.4f}")
            print("LDA transformer saved as 'lda_transformer.joblib'")

if __name__ == "__main__":
    main()
