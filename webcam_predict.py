import cv2
import numpy as np
import joblib
import os
import pyttsx3
import time
import threading

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    return resized.flatten().reshape(1, -1)

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'drowsiness_model.pkl'))
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    model = joblib.load(model_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return
    print("Press 'q' to quit.")

    drowsy_start_time = None
    warned = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        try:
            X = preprocess_frame(frame)
            pred = model.predict(X)[0]
            label = "alert" if pred == 0 else "drowsy"
        except Exception as e:
            label = "error"
            print(f"Prediction error: {e}")

        now = time.time()
        color = (0,255,0) if label == "alert" else (0,0,255)
        cv2.putText(frame, f"Status: {label.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if label == "drowsy":
            if drowsy_start_time is None:
                drowsy_start_time = now
                warned = False  # Reset warned flag on new drowsy episode
            elif not warned and now - drowsy_start_time >= 2:
                # Use threading to avoid blocking the video stream
                threading.Thread(target=speak, args=("Don't sleep! Wake up!",)).start()
                warned = True
        else:
            drowsy_start_time = None
            warned = False  # Reset when alert

        cv2.imshow("Driver Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()