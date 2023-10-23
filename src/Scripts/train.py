import argparse
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.utils import class_weight
from keras import layers, models

def create_model(model_num):
    if model_num == 1:
        model = models.Sequential([
            layers.Conv2D(16, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ])
    elif model_num == 2:
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
    else:
        raise ValueError("Invalid model number")
    
    return model

def compile_and_train(model, x_train, y_train, epochs, batch_size, validation_split, class_weights):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, class_weight=class_weights)

def save_model(model, model_path):
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)

def parse_args():
    parser = argparse.ArgumentParser(
        prog="Training Models",
        description="Choose your model and train it"
    )
    parser.add_argument('--data_path', help='Input data path')
    parser.add_argument('--model_path', help='Input model path')
    parser.add_argument('--m_num', type=int, help='Provide model number (1 or 2)')
    parser.add_argument('--blnc', help='Choose whether to balance the dataset or not')
    return parser.parse_args()

def main():
    args = parse_args()
    path_csv = args.data_path
    path_model = args.model_path
    model_num = args.m_num
    balance = args.blnc

    if model_num not in [1, 2]:
        print("Invalid model number")
        return

    df = pd.read_csv(path_csv)
    y_train = np.array(df['Label'])
    x_train = np.array(df['Image_Array'])

    if balance == "yes":
        c_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    else:
        c_weights = None

    model = create_model(model_num)
    compile_and_train(model, x_train, y_train, epochs=6, batch_size=32, validation_split=0.2, class_weights = c_weights)
    save_model(model, os.path.join(path_model, f'model_{model_num}_{balance}'))

if __name__ == '__main__':
    main()
