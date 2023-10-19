import numpy as np
import os
import sys
import csv
from Process_image import process_image

def load_images_labels(imgs, labs, path, label):
    for image_file in os.listdir(path):
        img = process_image(os.path.join(path, image_file))
        imgs.append(img)
        labs.append(label)

def labeling(imgs, labs, path):
    normal_path = os.path.join(path, 'NORMAL')
    pneumonia_path = os.path.join(path, 'PNEUMONIA')
    load_images_labels(imgs, labs, normal_path, 0)
    load_images_labels(imgs, labs, pneumonia_path, 1)


def main(path1, path2) :
    with open(path2, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Image_Array', 'Label'])  # Write header
        imgs, labels = [], []
        labeling(imgs, labels, path1)
        for image, label in zip(imgs, labels):
            image = image.flatten()  # Flatten the array if needed
            writer.writerow([image, label])


if __name__ == "__main__":
    input_data_directory = sys.argv[1]
    output_csv_file = sys.argv[2]
    main(input_data_directory, output_csv_file)