import os
from utils import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    X, y = load_dataset(data_dir)
    print(f"Total samples: {len(X)}, Total labels: {len(y)}")
    if len(X) == 0 or len(y) == 0:
        print("No data found. Please check your data folder and image files.")
        exit(1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["alert", "drowsy"]))
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'drowsiness_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")