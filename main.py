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
from CNN_keras import CNN

def main():
    cnn = CNN(use_gray = True, use_jitter = False)
    cnn.train(epochs=2,batch_size=32)
    pass
if __name__ == "__main__":
    main()




