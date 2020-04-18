import base64
import json
from io import BytesIO
import tensorflow as tf

import numpy as np
import requests
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image

# from flask_cors import CORS

app = Flask(__name__)

labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

# custom_labels = ['Daisy', 'Dandelion', 'Roses', 'Sunflowers', 'Tulips']

# Uncomment this line if you are making a Cross domain request
# CORS(app)

# Testing URL
@app.route('/hello/', methods=['GET', 'POST'])
def hello_world():
    return 'Hello, World!'


@app.route('/imageclassifier/predict/', methods=['POST'])
def image_classifier():
    # Decoding and pre-processing base64 image
    # img = image.img_to_array(image.load_img(BytesIO(base64.b64decode(request.form['b64'])),
    #                                         target_size=(224, 224))) / 255.

    img = image.img_to_array(image.load_img(
        request.files['image'], target_size=(224, 224, 3))) / 255

    # this line is added because of a bug in tf_serving < 1.11
    img = img.astype('float16')

    # Creating payload for TensorFlow serving request
    payload = {
        "instances": [{'input_image': img.tolist()}]
    }

    data = json.dumps({"signature_name": "serving_default",
                       "instances": [img.tolist()]})

    # Making POST request
    r = requests.post(
        'http://localhost:9000/v1/models/ImageClassifier:predict', data=data)

    print(r.content)
    obj = json.loads(r.content.decode('utf-8'))
    obj_pred = obj['predictions']
    # print(obj_pred[0])
    predicts = np.array(obj_pred[0])
    top_three = np.argsort(predicts)[-3:][::-1]
    # print(imagenet_labels[top_three])

    # Decoding results from TensorFlow Serving server
    # pred = json.loads(np.array2string(CLASS_NAMES[top_three], separator=','))
    #     pred = json.loads(r.content.decode('utf-8'))

    # Returning JSON response to the frontend
    return np.array2string(imagenet_labels[top_three], separator=',')
