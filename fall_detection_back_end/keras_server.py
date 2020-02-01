# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 00:56:31 2020

@author: umesh
"""

# keras_server.py

# Python program to expose a ML model as flask REST API

# import the necessary modules
from keras.models import load_model
import tensorflow as tf
import numpy as np
import flask
import json
import winsound
import time

# Create Flask application and initialize Keras model
app = flask.Flask(__name__)

# Function to Load the model
def start_load_model():

    # global variables, to be used in another function
    global doppler_model
    doppler_model = load_model('classifier_final.sav')
    global doppler_graph
    doppler_graph = tf.get_default_graph()
    global doppler_accuracy
    global doppler_time_stamp


    global android_model
    android_model = load_model('android_classifier_final.sav')
    global android_graph
    android_graph = tf.get_default_graph()
    global android_accuracy
    global android_time_stamp

    global trigger_fall_time
    trigger_fall_time = 0


@app.route("/predict/doppler", methods =["POST"])
def dopplerPredict():
    data = {} # dictionary to store result
    data["success"] = False
    # Check if image was properly sent to our endpoint
    if flask.request.method == "POST":

        request = flask.request
        reading_counts = request.get_json()
        reading_counts = reading_counts['data']
        if reading_counts is not None:
            reading_counts = np.asarray(reading_counts).reshape(1,5)
            # Predict ! global preds, results
            with doppler_graph.as_default():
                preds = doppler_model.predict(reading_counts)
                data["prediction"] = format(preds[0][0], '.2f')
                if float(data["prediction"]) >= 0.5: 
                    print("Doppler Fall Detected")
                    playNoise()
            data["success"] = True
    # return JSON response
    return json.dumps(str(data))

@app.route("/predict/android", methods =["POST"])
def androidPredict():
    data = {} # dictionary to store result
    data["success"] = False

    # Check if image was properly sent to our endpoint
    if flask.request.method == "POST":
        request = flask.request
        reading_counts = request.get_json()
        avm = int(reading_counts['avm'])
        reading_counts = reading_counts['data']
        if reading_counts is not None:
            reading_counts = np.asarray(reading_counts).reshape(1,4)
            # Predict ! global preds, results
            with android_graph.as_default():
                preds = android_model.predict(reading_counts)
                data["prediction"] = format(preds[0][0], '.2f')
            data["success"] = True
            if avm == 1 or float(data["prediction"]) >= 0.5:
                print("Android Fall Detected")
                playNoise()
    # return JSON response
    return flask.jsonify(data)

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def playNoise():
    curr_time = int(time.time()*1000.0)
    global trigger_fall_time
    if curr_time - trigger_fall_time < 2000:
        print("Ignoring ALarm !!")
    trigger_fall_time = curr_time        
    winsound.Beep(1000, 2000)

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    start_load_model()
    app.run(host='0.0.0.0' , port=5000)
