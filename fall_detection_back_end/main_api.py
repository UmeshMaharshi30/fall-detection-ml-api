# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 17:08:05 2020

@author: umesh
"""

import flask
from flask import request

from keras.models import load_model
import numpy as np

app = flask.Flask(__name__)


android_sensor = 0;
doppler_sensor = 0;

android_acc = 0;
doppler_acc = 0;

@app.route('/', methods=['GET'])
def home():
    return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"

@app.route('/android', methods=['POST'])
def fall_android():
    print(request.form.getlist('data'))
    if request.form.getlist('data') is None:
        return "Empty"
    #input_list = np.asarray(request.form.getlist('data[]')).reshape(1,4)
    #prediction = doppler_classifier.predict(input_list)
    #prediction = prediction[0][0]
    #print(prediction)
    return "fall"


@app.route('/doppler', methods=['POST'])
def fall_doppler():
    #input_list = request.get_json()['data']
    doppler_classifier = load_model('classifier_final.sav')
    input_list = [110, 28, 20, 52, 0.03]
    temp_list = np.asarray(input_list).astype(object).reshape(1,5)
    print(temp_list)
    prediction = doppler_classifier.predict(temp_list)
    prediction = prediction[0][0]
    print(prediction)
    return "hello"


app.run(debug = False)