import os
import cv2
import numpy as np
import joblib

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = cv2.resize(img, (64, 64))
    return img.flatten().reshape(1, -1)

if __name__ == "__main__":
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'drowsiness_model.pkl'))
    print("Looking for model at:", model_path)
    if not os.path.exists(model_path):
        print("Model file not found at:", model_path)
        exit(1)
    model = joblib.load(model_path)
    test_image_path = input("Enter path to image for prediction: ").strip()
    if not os.path.exists(test_image_path):
        print(f"Image file not found: {test_image_path}")
        exit(1)
    try:
        X_test = preprocess_image(test_image_path)
    except Exception as e:
        print("Error processing image:", e)
        exit(1)
    pred = model.predict(X_test)[0]
    label = "alert" if pred == 0 else "drowsy"
    print(f"The model predicts: {label}")