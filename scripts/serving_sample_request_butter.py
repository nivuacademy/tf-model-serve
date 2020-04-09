import argparse
import json

import os
import cv2
import numpy as np
import requests
from keras.applications import inception_v3


classes = ['Lime Butterfly',
 'Southern Birdwing',
 'Malabar Raven',
 'Tailed Jay',
 'Malabar Banded Peacock',
 'Common Rose',
 'Common Banded Peacock',
 'Blue Mormon',
 'Spot Swordtail',
 'Common Mime',
 'Common Mormon',
 'Red Helen',
 'Malabar Banded Swallowtail',
 'Five-bar Swordtail']

image_path = '../test_images/car.jpg'
IMG_SIZE = 224

img_array = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)

# Preprocessing our input image
img = np.array(new_array) / 255.

# this line is added because of a bug in tf_serving(1.10.0-dev)
img = img.astype('float16')

payload = {
    "instances": [{'input_image': img.tolist()}]
}

# sending post request to TensorFlow Serving server
r = requests.post('http://34.93.104.79:9000/v1/models/ButterClassifier:predict', json=payload)
pred = json.loads(r.content.decode('utf-8'))

# print(pred)

# pvals = np.array(pred['predictions'][0])

# c = pvals.argmax()
# print("index", c)
# print("class", classes[c])

# Decoding the response
# decode_predictions(preds, top=5) by default gives top 5 results
# You can pass "top=10" to get top 10 predicitons
print(json.dumps(inception_v3.decode_predictions(np.array(pred['predictions']))[0]))
