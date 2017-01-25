import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import matplotlib.gridspec as gridspec
import csv
import random
import os
import shutil
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split
import pandas as pd
from model import Model
from tqdm import tqdm
from sklearn.utils import shuffle
from time import localtime, strftime

import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K


class CNN():
    def __init__(self, use_gray=False, use_jitter=False):
        self.use_gray = use_gray
        self.use_jitter = use_jitter
        self.img_rows, self.img_cols = 32, 32
        if self.use_gray:
            self.input_shape = (self.img_rows, self.img_cols)
        else:
            self.input_shape = (self.img_rows, self.img_cols, 3)
        self.nb_classes = 43
        self.samplesPerTrack = {}
        self.X_train_paths, self.X_val_paths, self.y_train, self.y_val = [], [], [], []
        self.labelDict = self.readLabelFromFile()
        self.df_train = pd.read_csv("./train_data.csv", sep=",\s*", engine='python')
        self.df_jitter = pd.read_csv("./jitter_data.csv", sep=",\s*", engine='python')
        self.df_test = pd.read_csv("./test_data.csv", sep=",\s*", engine='python')
        self.X_test_paths = list(self.df_test['path'])
        self.y_test = list(self.df_test['y'])
        self.encoder = LabelBinarizer()
        self.encoder.fit(list(self.df_train["y"]))
        self.cat()
        self.split()
        self.model = self.getModel()

    def getModel(self):
        pool_size = (2, 2)
        model = Sequential()
        if self.use_gray:
            model.add(Reshape((self.img_cols, self.img_rows, 1), input_shape=self.input_shape))
        else:
            model.add(Reshape((self.img_cols, self.img_rows, 3), input_shape=self.input_shape))
        model.add(Convolution2D(8, 3, 3,
                                border_mode='same',
                                subsample=(1, 1),
                                name="conv1"))
        model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=pool_size,border_mode="same"))
        # model.add(Dropout(0.7))

        model.add(Convolution2D(16, 3, 3,
                                border_mode='same',
                                subsample=(1, 1),
                                name="conv2"))
        model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=pool_size,border_mode="same"))
        # model.add(Dropout(0.7))

        model.add(Convolution2D(24, 3, 3,
                                border_mode='same',
                                subsample=(2, 2),
                                name="conv3"))
        model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2,2),border_mode="same"))
        model.add(Dropout(0.7))

        model.add(Convolution2D(32, 3, 3,
                                border_mode='same',
                                subsample=(1, 1),
                                name="conv4"))
        model.add(Activation('relu'))
        model.add(Convolution2D(48, 3, 3,
                                border_mode='same',
                                subsample=(1, 1),
                                name="conv5"))
        model.add(Activation('relu'))
        model.add(Convolution2D(56, 3, 3,
                                border_mode='same',
                                subsample=(2, 2),
                                name="conv6"))
        model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2,2),border_mode="same"))
        model.add(Dropout(0.7))

        model.add(Flatten(name="flat"))
        model.add(Dense(128, input_dim=model.get_layer(name="flat").output_shape[1], name="hidden1"))
        model.add(Activation('relu'))
        model.add(Dropout(0.7))
        model.add(Dense(43, input_dim=128, name="output"))
        model.add(Activation('softmax', name="softmax"))
        opt = keras.optimizers.Adagrad(lr=1e-5, epsilon=1e-08, decay=0.0)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        print(model.summary())
        return model

    def decode(self, Y):
        id = self.encoder.inverse_transform(np.array([Y]))
        return id, self.labelDict[str(id[0])]

    def readLabelFromFile(self):
        labelDict = {}
        with open('signnames.csv', 'r') as csvfile:
            content = csv.reader(csvfile, delimiter=',')
            for row in content:
                labelDict[row[0]] = row[1]
        return labelDict

    def cat(self):
        print("Categorizing...")
        tracks = self.df_train['y'].unique()  # a list of unique tracks
        self.samplesPerTrack = {}
        for track in tracks:
            self.samplesPerTrack[track] = []
            self.samplesPerTrack[track].extend(list(self.df_train[self.df_train['y'] == track]['path']))
        if self.use_jitter:
            for track in tracks:
                self.samplesPerTrack[track].extend(list(self.df_jitter[self.df_jitter['y'] == track]['path']))
        print("Categorization completes.")

    def imgPreprocess(self, x):
        """
        :param x: one single image
        :return:
        """
        if self.use_gray:
            ret = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        else:
            ret = x
        ret = ret.astype('float64')
        ret -= 128
        ret /= 255
        return ret

    def split(self):
        print("Splitting...")
        for track in self.samplesPerTrack.keys():
            paths = self.samplesPerTrack[track]
            labels = [track for i in range(len(paths))]
            paths_train, paths_val, labels_train, labels_val = train_test_split(paths, labels, test_size=0.2,
                                                                                random_state=42)
            self.X_train_paths.extend(paths_train)
            self.X_val_paths.extend(paths_val)
            self.y_train.extend(labels_train)
            self.y_val.extend(labels_val)
        print("Train #={}".format(len(self.y_train)))
        print("Val #={}".format(len(self.y_val)))
        print("Splitting completes.")

    def generate_from_directory(self, X_paths, y, batch_size=32, break_if_finish=False):
        """
        :param X_path: paths to stored images (list)
        :param y: target angles
        :return: image in the form of numpy arrays; target values in the form of np.array
        """
        assert len(X_paths) == len(y)
        Y = self.encoder.transform(y)
        numOfBatches = int(Y.shape[0] / batch_size)
        start_index = 0
        while 1:
            if start_index + batch_size <= len(X_paths):
                X = []
                for j in range(batch_size):
                    x = cv2.imread(X_paths[j])
                    x = self.imgPreprocess(x)
                    X.append(x)
                X = np.array(X)
                yield (X, Y[start_index: start_index + batch_size, :])
                start_index = start_index + batch_size
            else:
                X = []
                for j in range(len(X_paths) - start_index):
                    x = cv2.imread(X_paths[j])
                    x = self.imgPreprocess(x)
                    X.append(x)
                X = np.array(X)
                yield (X, Y[start_index: len(X_paths), :])
                if break_if_finish:
                    return
                start_index = 0

    def train(self, epochs, batch_size):
        history = self.model.fit_generator(self.generate_from_directory(self.X_train_paths, self.y_train, batch_size=batch_size),
                                 samples_per_epoch=len(self.X_train_paths), nb_epoch=epochs,
                                 validation_data=self.generate_from_directory(self.X_val_paths, self.y_val),
                                 nb_val_samples=len(self.y_val))
        print("val_acc history:")
        print(history.history["val_acc"])
        res = self.model.evaluate_generator(self, self.generate_from_directory(self.X_test_paths, self.y_test, batch_size=batch_size), \
                           val_samples=len(self.y_test))
        print("Test Acc:", str(res[1]))
