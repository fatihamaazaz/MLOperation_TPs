import flask
from flask import jsonify, render_template, request
import pickle
import numpy as np
from keras.preprocessing import image

app = flask.Flask(__name__)
app.config["debug"] = True

def process_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the pixel values
    return img

@app.route("/")
def home():
    # render landing page
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    model_file_path = r"C:\Users\DESKTOP-FM\Desktop\MLOps TP1\MLOperation_TPs\src\Models\Model_1.pckl"
    with open(model_file_path, 'rb') as model_file:
        model = pickle.load(model_file)

    image = request.files["image"]
    if image:
        image_path = "path_to_save_uploaded_image.jpg"  # Adjust the path to where you want to save the image
        image.save(image_path)
        img = process_image(image_path)
        pred = model.predict(img)
        prediction = pred[0][0]
        result = 'pneumonia' if prediction > 0.5 else 'normal'
        response = {'prediction': result}
        return jsonify(response)
    else:
        return jsonify({'error': 'No image provided'})

if __name__ == "__main__":
    app.run(host="0.0.0.0")
