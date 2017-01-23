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

class CNN():
    def __init__(self):
        self.labelDict = self.readLabelFromFile()
        self.img_rows, self.img_cols = 32, 32
        self.nb_classes = 43

    def readLabelFromFile(self):
        labelDict = {}
        with open('signnames.csv', 'r') as csvfile:
            content = csv.reader(csvfile, delimiter=',')
            for row in content:
                labelDict[row[0]] = row[1]
        return labelDict

    def grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def transform_image(img, ang_range, shear_range, trans_range):
        '''
        This function transforms images to generate new images.
        The function takes in following arguments,
        1- Image
        2- ang_range: Range of angles for rotation
        3- shear_range: Range of values to apply affine transform to
        4- trans_range: Range of values to apply translations over.
        A Random uniform distribution is used to generate different parameters for transformation
        '''
        # Rotation
        ang_rot = np.random.uniform(ang_range) - ang_range / 2
        rows = img.shape[0]
        cols = img.shape[1]
        Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)

        # Translation
        tr_x = trans_range * np.random.uniform() - trans_range / 2
        tr_y = trans_range * np.random.uniform() - trans_range / 2
        Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

        # Shear
        pts1 = np.float32([[5, 5], [20, 5], [5, 20]])
        pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
        pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2
        pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
        shear_M = cv2.getAffineTransform(pts1, pts2)

        img = cv2.warpAffine(img, Rot_M, (cols, rows))
        img = cv2.warpAffine(img, Trans_M, (cols, rows))
        img = cv2.warpAffine(img, shear_M, (cols, rows))

        if (len(img.shape) == 2):
            img = img.reshape(img.shape[0], img.shape[1], 1)
        elif (len(img.shape) == 3):
            pass
        else:
            raise ValueError("Not Supported image shape!")
        return img

    def cat(self, X, y):
        """
        Return a list instead of numpy array!
        """
        samplesPerTrack = {}
        for i in range(X.shape[0]):
            if (not y[i] in samplesPerTrack.keys()):
                samplesPerTrack[y[i]] = []
                samplesPerTrack[y[i]].append(X[i])
            else:
                samplesPerTrack[y[i]].append(X[i])
        return samplesPerTrack

    def GrayNormResize(self,X):
        X_ret = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
        for i in range(X.shape[0]):
            X_ret[i] = self.grayscale(X[i])
        X_ret = X_ret.astype('float64')
        X_ret -= 128
        X_ret /= 255
        X_ret1 = np.reshape(X_ret, (X_ret.shape[0], X_ret.shape[1], X_ret.shape[2], 1))
        return X_ret1

    def jitter(self, X, y, jitter_ratio=2):
        """
        Take X, y as numpy array and return numpy arrays X_ret, y_ret
        """
        sampleBasedOnTrack = cat(X, y)
        # Find Max Freq
        max_freq = 0
        for i in sampleBasedOnTrack.keys():
            if max_freq < len(sampleBasedOnTrack[i]):
                max_freq = len(sampleBasedOnTrack[i])
        print("max_freq = " + str(max_freq))
        for i in sampleBasedOnTrack.keys():
            count = len(sampleBasedOnTrack[i])
            if count < max_freq * jitter_ratio:
                numToInsert = int(max_freq * jitter_ratio - count)
                for j in range(numToInsert):
                    transformed = transform_image(random.choice(sampleBasedOnTrack[i]), 10, 4, 4)
                    sampleBasedOnTrack[i].append(transformed)
        for i in sampleBasedOnTrack.keys():
            # print("sampleBasedOnTrack [{}] len".format(i)+str(len(sampleBasedOnTrack[i])))
            # print("sampleBasedOnTrack[0][0].shape:"+str(sampleBasedOnTrack[0][0].shape))
            sampleBasedOnTrack[i] = np.array(sampleBasedOnTrack[i])
        X_ret = []
        y_ret = []
        for i in sampleBasedOnTrack.keys():
            data = sampleBasedOnTrack[i]
            X_ret.extend(data)
            y_ret.extend([i for j in range(len(data))])
        X_ret = np.array(X_ret)
        y_ret = np.array(y_ret)
        return X_ret, y_ret

    def split(self, X_train, y_train):
        samplesPerTrack = self.cat(X_train, y_train)
        X_train_split = []
        X_val_split = []
        y_train_split = []
        y_val_split = []
        for track in samplesPerTrack.keys():
            data = samplesPerTrack[track]
            labels = [track for i in range(len(data))]
            data_train, data_val, labels_train, labels_val = train_test_split(data, labels, test_size=0.2,
                                                                              random_state=42)
            X_train_split.extend(data_train)
            X_val_split.extend(data_val)
            y_train_split.extend(labels_train)
            y_val_split.extend(labels_val)
        X_train_split = np.array(X_train_split)
        X_val_split = np.array(X_val_split)
        y_train_split = np.array(y_train_split)
        y_val_split = np.array(y_val_split)
        return X_train_split, y_train_split, X_val_split, y_val_split

    def readData(self):
        training_file = "./train.p"
        testing_file = "./test.p"
        with open(training_file, mode='rb') as f:
            train = pickle.load(f)
        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)
        X_train, y_train = train['features'], train['labels']
        X_test, y_test = test['features'], test['labels']
        assert (X_train.shape[0] == y_train.shape[
            0]), "number of images!=number of labels."
        assert (X_train.shape[1:] == (32, 32, 3)), "Image dimention must be 32 x 32 x 3."
        encoder = LabelBinarizer()
        encoder.fit(y_train)
        return X_train, y_train, X_test, y_test, encoder


    def preprocess(self):
        X_train, y_train, X_test, y_test, encoder = self.readData()
        self.split(X_train, y_train)
