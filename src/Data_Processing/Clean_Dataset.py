import os
import cv2
import sys


# Function to check if an image is corrupted
def is_image_corrupted(image_path):
    try:
        # Attempt to open and read the image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return True
        return False
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return True

def main(path) :
    if not os.path.exists(path):
        print(f"Folder not found: {path}")
    for filename in os.listdir(path):
        image_path = os.path.join(path, filename)
        # Check if the image is corrupted
        if is_image_corrupted(image_path):
           os.remove(image_path)


if __name__ == "__main__":
    path = sys.argv[1]
    main(path)
