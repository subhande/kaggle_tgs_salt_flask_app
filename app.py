
import os
import numpy as np
import shutil

from PIL import Image
import io
import base64


from keras.models import load_model
import cv2
import tensorflow as tf
from keras import backend as K
import keras

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
session = tf.Session(config=config)
graph = tf.get_default_graph()
keras.backend.set_session(session)


# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Model saved with Keras model.save()
MODEL_PATH = 'models/model_tgs_salt_1.h5'


# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# Load your trained model
model = load_model(MODEL_PATH, custom_objects={"mean_iou": mean_iou})
model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# print('Model loaded. Check http://127.0.0.1:5000/')


"""
    This function takes a file and generates test data
"""
def generate_test_dataset(file, image_size):
    X_test = []
    img = np.float32(cv2.resize(cv2.imread(file, 1), (128, 128))) / 255.0
    X_test.append(img)
    return np.asarray(X_test)


"""
    This function takes filenames and returns output image
"""

def model_predict(img_path, model):
    with session.as_default():
        with session.graph.as_default():
            X_test = generate_test_dataset(img_path, (128, 128, 1))

            preds_test = model.predict(X_test)
            preds_test_t = (preds_test > 0.5).astype(np.uint8)
            return cv2.resize(preds_test_t[0] * 255, (101, 101))

# Redering home page
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


# Predict Route
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        basepath = os.path.dirname(__file__)

        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # return image in base64 format
        img = Image.fromarray(preds.astype('uint8'))
        # print(img)
        rawBytes = io.BytesIO()
        img.save(rawBytes, "PNG")
        rawBytes.seek(0)
        img_base64 = base64.b64encode(rawBytes.read())
        shutil.rmtree(os.path.join(basepath, 'uploads'))
        os.mkdir(os.path.join(basepath, 'uploads'))

        return jsonify({'image':str(img_base64)})

    return jsonify({"status": "some error occured while sending file"}), 500


# if __name__ == '__main__':
#     app.run()

# RUN FLASK APPLICATION
if __name__ == '__main__':

    # RUNNNING FLASK APP
    app.run(host = '0.0.0.0', port=8080)

