import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tqdm import tqdm
import Utils

SIGNATURE_SIZE = 128
SIGNATURE_MODEL_PATH = '/home/hoangcs/da_orc_research/Soucre/Team01/saved_model/signature_model' 

def DataLoader(imagePath):
    data = []
    #imageArr=plt.imread(imagePath)
    #imageArr = imagePath
    #imageArr=imageArr[...,::1]
    imagePath = Utils.Img_To_Ndarray(imagePath)
    resizedArr = cv2.resize(imagePath, (SIGNATURE_SIZE, SIGNATURE_SIZE))
    data.append(resizedArr)
    return np.array(data)

def SignDect(signatureImage):
    inputSet=DataLoader(signatureImage)
    new_model = tf.keras.models.load_model(SIGNATURE_MODEL_PATH)
    prediction = new_model.predict(inputSet)
    res = np.argmax(prediction, axis =1)
    return int(res)


