import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import matplotlib.gridspec as gridspec
import csv
import random
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split
#from classifer import CNN
#from CNN_keras import CNN
#from CNN_keras_debug import CNN
#from CNN_from_nparray import CNN
from CNN import CNN

def main():
    cnn = CNN(use_gray = True, use_jitter = True)
    #cnn.test_saved_model(model_path="./log/model.ckt")
    #img = cv2.imread("./myimg/stop.jpeg")
    #cnn.predict(img,"./log/model.ckt")
    cnn.train(epochs=50,batch_size=256)
if __name__ == "__main__":
    main()




