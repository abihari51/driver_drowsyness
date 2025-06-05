import os
import cv2
import numpy as np

def load_images_from_folder(folder, label):
    images, labels = [], []
    if not os.path.exists(folder):
        print(f"Folder does NOT exist: {folder}")
        return images, labels
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                images.append(img.flatten())
                labels.append(label)
    return images, labels

def load_dataset(base_dir):
    X, y = [], []
    for label, name in enumerate(['alert', 'drowsy']):
        folder = os.path.join(base_dir, name)
        Xi, yi = load_images_from_folder(folder, label)
        print(f"Loaded {len(Xi)} images for class '{name}'")
        X.extend(Xi)
        y.extend(yi)
    return np.array(X), np.array(y)