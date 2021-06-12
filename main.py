
import sys
import os
import glob
import re
import numpy as np

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)

MODEL_PATH = "model.h5"
model = tf.keras.models.load_model('model.h5')
print('Model loaded. Check http://127.0.0.1:5000/ or http://localhost:5000/')

def model_predict(img_path, model):
    IMG_SIZE = 224
    img_array = cv2.imread(img_path)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE), 3)
    new_array = new_array.reshape(1, 224, 224, 3)
    prediction = model.predict([new_array])
    return prediction


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')
    # return "Hello World"

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join( basepath, 'uploads', secure_filename(f.filename) )
        f.save(file_path)

        # Make prediction
        prediction = model_predict(file_path, model)

        # These are the prediction categories 
        CATEGORIES = ['CANCER', 'NORMAL']
        
        # getting the prediction result from the categories
        result = CATEGORIES[int(round(prediction[0][0]))]
        
        # returning the result
        return result
    
    # if not a 'POST' request we then return None
    return None


if __name__ == '__main__':
    app.run(debug=True)

  




