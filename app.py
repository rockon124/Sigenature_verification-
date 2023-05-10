

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer
import pickle
import cv2
# Define a flask app
app = Flask(__name__)

CATEGORIES = ["Fake","ORIGINAL"]

signatureRF = pickle.load(open('signature-rf.pkl','rb'))

def model_predict(img_path, model):
    
    def random_forest_predictions(testX, signatureRF):
        predictions = signatureRF.predict(testX)    
        return predictions

    img_array = cv2.imread(img_path)
    new_array = cv2.resize(img_array, (100, 100))
    new_array = cv2.cvtColor(new_array, cv2.COLOR_BGR2GRAY).reshape(10000)

    predictedIndex = random_forest_predictions([new_array], signatureRF)

    preds = CATEGORIES[signatureRF.classes_[predictedIndex[0]]]

    print("THE SIGNATURE is",preds)
    return preds
    #return render_template('index.html', preds = preds)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, signatureRF)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run()
