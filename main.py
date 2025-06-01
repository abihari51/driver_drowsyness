import argparse
import os

def run_train():
    os.system("python src/train.py")

def run_predict():
    os.system("python src/predict.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver Drowsiness Detection")
    parser.add_argument("--mode", choices=["train", "predict"], required=True, help="train or predict")
    args = parser.parse_args()
    if args.mode == "train":
        run_train()
    elif args.mode == "predict":
        run_predict()