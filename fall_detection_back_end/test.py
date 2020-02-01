import numpy as np
from keras.models import load_model

def prediction(input_list):
    input_list = [110, 28, 20, 52, 0.03]
    doppler_classifier = load_model('classifier_final.sav')
    
    temp_list = np.asarray(input_list).astype(object).reshape(1,5)
    print(temp_list)
    prediction = doppler_classifier.predict(temp_list)
    prediction = prediction[0][0]
    print(prediction)


prediction([])