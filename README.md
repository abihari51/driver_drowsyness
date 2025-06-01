# Driver Drowsiness Detection (ML Approach)

## Overview
A simple Python project to detect driver drowsiness using classical machine learning and OpenCV for face detection.

## Project Structure

```
driver-drowsiness-ml/
│
├── data/
│   ├── alert/
│   └── drowsy/
├── model/
│   └── drowsiness_model.pkl
├── src/
│   ├── collect_data.py
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── requirements.txt
├── README.md
└── main.py
```

## Setup

1. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

2. **Collect Data**
   - Run the data collection script and press `c` to capture images, `q` to quit.
     ```
     python src/collect_data.py
     ```
   - Collect both "alert" and "drowsy" images.

3. **Preprocess Data**
   ```
   python src/preprocess.py
   ```

4. **Train the Model**
   ```
   python main.py --mode train
   ```

5. **Run Real-Time Detection**
   ```
   python main.py --mode predict
   ```

## Notes

- Make sure your webcam works with OpenCV.
- The `data/alert` and `data/drowsy` folders should contain your images for each class.
- For better accuracy, collect diverse images under different lighting and angles.

## License

MIT