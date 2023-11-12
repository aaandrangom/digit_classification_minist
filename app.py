from flask import Flask, render_template, request, jsonify
import numpy as np
from keras.models import load_model
import imageio

app = Flask(__name__)

# Carga el modelo Keras desde el archivo .h5
model = load_model('model_Mnist.h5')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    image_file = request.files['file']

    if image_file.filename == '':
        return jsonify({'error': 'No selected file'})

    if image_file:
        # Leer la imagen y realizar preprocesamiento
        im = imageio.imread(image_file)
        gray = np.dot(im[..., :3], [0.299, 0.587, 0.114])
        gray = gray.reshape(1, 28, 28, 1) / 255.0

        # Realizar la predicción
        prediction = model.predict(gray)
        predicted_number = prediction.argmax()

        return jsonify({'predicted_number': int(predicted_number)})


if __name__ == '__main__':
    app.run(debug=True)
