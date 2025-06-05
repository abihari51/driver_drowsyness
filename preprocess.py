import cv2
import os

def preprocess_image(image_path, output_size=(64, 64)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, output_size)
    return img

def preprocess_folder(input_folder, output_folder, output_size=(64, 64)):
    os.makedirs(output_folder, exist_ok=True)
    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        img = preprocess_image(img_path, output_size)
        if img is not None:
            cv2.imwrite(os.path.join(output_folder, img_name), img)

if __name__ == "__main__":
    for label in ["alert", "drowsy"]:
        input_folder = f"../data/{label}"
        output_folder = f"../data/{label}_processed"
        preprocess_folder(input_folder, output_folder)