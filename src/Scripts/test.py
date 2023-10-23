from sklearn.metrics import classification_report
import pickle
import argparse
import pandas as pd
import numpy as np

def test(model, x_test, y_test):
    baseline_metrics = model.evaluate(x_test, y_test)
    y_pred_baseline = (model.predict(x_test) > 0.5).astype(int)
    class_report = classification_report(y_test, y_pred_baseline)

    print("Classification Report:\n", class_report)


def load_model(model_file_path):
    with open(model_file_path, 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    return loaded_model

def parse_args():
    parser = argparse.ArgumentParser(
        prog="Test Models",
        description="Choose a model and test it"
    )
    parser.add_argument('--data_path', help='Input data path')
    parser.add_argument('--model_path', help='Input model path')
    return parser.parse_args()

def main():
    args = parse_args()
    path_csv = args.data_path
    path_model = args.model_path
    model = load_model(path_model)
    df = pd.read_csv(path_csv)
    y_test = np.array(df['Label'])
    x_test = np.array(df['Image_Array'])

    test(model, x_test, y_test)


if __name__ == '__main__':
    main()
