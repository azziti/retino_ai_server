from posixpath import dirname, join, realpath
import numpy as np
import pandas as pd
import os


from PIL import Image

from tensorflow.keras.applications.imagenet_utils import (decode_predictions,
                                                          preprocess_input)
from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.models import load_model





from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input




from tensorflow.nn import softmax
from numpy import argmax
from numpy import max
from numpy import array



def load_model2():
    """
    Loads and returns the pretrained model
    """
 
   
     
    #print("Model loaded")
    return model


def prepare_image(image, target):

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
   # image = np.expand_dims(image, axis=0)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))  # Créer la collection d'images (un seul échantillon)
    image = preprocess_input(image)  # The preprocess_input function is meant to adequate your image to the format the model requires.

   # img = load_img('testRD.jpg', target_size=(64, 64))  # Charger l'image
   # img = img_to_array(img)  # Convertir en tableau numpy
    # img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))  # Créer la collection d'images (un seul échantillon)
    # img = preprocess_input(img)  # Prétraiter l'image comme le veut VGG-16

    return image


def predict(image, model):
    # We keep the 2 classes with the highest confidence score
    # results = decode_predictions(model.predict(image), 2)[0]   # TODO : fix it
    class_predictions = array([
    '0',
    '1',
    '2',
    '3',
    '4'
    
])

    pred = model.predict(image)
    score = softmax(pred[0])

    class_prediction = class_predictions[argmax(score)]
    model_score = round(max(score) * 100, 2)



    response = [
        {"class": class_prediction, "score": model_score} 
    ]
    return response

















































