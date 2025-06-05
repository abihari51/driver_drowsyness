import cv2
import os

def collect_images(label, save_dir, num_samples=100):
    cap = cv2.VideoCapture(0)
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    print(f"Collecting images for label: {label}")
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow(f"Collecting {label}", frame)
        key = cv2.waitKey(1)
        if key == ord('c'):
            img_path = os.path.join(save_dir, f"{label}_{count}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Saved: {img_path}")
            count += 1
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    label = input("Enter label (alert/drowsy): ")
    save_dir = f"../data/{label}"
    collect_images(label, save_dir)