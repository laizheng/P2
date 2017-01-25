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

class CNN():
    def __init__(self, use_gray = False, use_jitter = False):
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
        self.df_test = pd.read_csv("./test_data.csv", sep=",\s*",engine='python')
        self.X_test_paths = list(self.df_test['path'])
        self.y_test = list(self.df_test['y'])
        self.encoder = LabelBinarizer()
        self.encoder.fit(list(self.df_train["y"]))
        self.cat()
        self.split()
        self.model = Model(input_shape=self.input_shape)

    def decode(encoder, label, labelDict):
        id = encoder.inverse_transform(np.array([label]))
        return id, labelDict[str(id[0])]

    def readLabelFromFile(self):
        labelDict = {}
        with open('signnames.csv', 'r') as csvfile:
            content = csv.reader(csvfile, delimiter=',')
            for row in content:
                labelDict[row[0]] = row[1]
        return labelDict

    def cat(self):
        print("Categorizing...")
        tracks = self.df_train['y'].unique() # a list of unique tracks
        self.samplesPerTrack = {}
        for track in tracks:
            self.samplesPerTrack[track] = []
            self.samplesPerTrack[track].extend(list(self.df_train[self.df_train['y'] == track]['path']))
        if self.use_jitter:
            for track in tracks:
                self.samplesPerTrack[track].extend(list(self.df_jitter[self.df_jitter['y'] == track]['path']))
        print("Categorization completes.")

    def imgPreprocess(self,x):
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
            paths_train, paths_val, labels_train, labels_val = train_test_split(paths, labels, test_size=0.2,random_state=42)
            self.X_train_paths.extend(paths_train)
            self.X_val_paths.extend(paths_val)
            self.y_train.extend(labels_train)
            self.y_val.extend(labels_val)
        print("Train #={}".format(len(self.y_train)))
        print("Val #={}".format(len(self.y_val)))
        print("Splitting completes.")

    def generate_from_directory(self, X_paths, y, batch_size=32, break_if_finish = False):
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
                yield (X, Y[start_index: start_index + batch_size,:])
                start_index = start_index + batch_size
            else:
                X = []
                for j in range(len(X_paths) - start_index):
                    x = cv2.imread(X_paths[j])
                    x = self.imgPreprocess(x)
                    X.append(x)
                X = np.array(X)
                yield (X, Y[start_index: len(X_paths),:])
                if break_if_finish:
                    return
                start_index = 0

    def getValAccuracy(self,sess):
        print("computing Val Acc..."),
        acc = []
        for X, Y in tqdm(self.generate_from_directory(self.X_val_paths,self.y_val,batch_size=32,break_if_finish=True), \
                         desc='Val Acc'):
            acc.append(sess.run(self.model.accuracy,feed_dict={self.model.X:X, self.model.Y_true:Y}))
        return np.mean(acc)

    def getTestAccuracy(self,sess):
        print("computing Test Acc..."),
        acc = []
        for X, Y in tqdm(self.generate_from_directory(self.X_test_paths,self.y_test,batch_size=32,break_if_finish=True), \
                         desc='Test Acc'):
            acc.append(sess.run(self.model.accuracy,feed_dict={self.model.X:X, self.model.Y_true:Y}))
        return np.mean(acc)

    def train(self,epochs,batch_size):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        log_batch_step = 100
        batch_num = []
        cost_batch = []
        val_acc_epoch = []
        test_acc_epoch = []
        print("Training Begin...")
        for epoch_i in range(epochs):
            self.X_train_paths, self.y_train = shuffle(self.X_train_paths,self.y_train)
            gen = self.generate_from_directory(self.X_train_paths, self.y_train, batch_size=batch_size)
            batch_count = int(len(self.X_train_paths)/batch_size) + 1
            batches_pbar = range(batch_count)
            batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i + 1, epochs),leave=False)
            for batch_cnt in batches_pbar:
                X, Y = next(gen)
                # Run optimizer and get loss
                _, l = sess.run(
                    [self.model.optimizer, self.model.cost],
                    feed_dict={self.model.X:X, self.model.Y_true:Y})
                # Log every 50 batches
                if not batch_cnt % log_batch_step:
                    previous_batch = batch_num[-1] if batch_num else 0
                    batch_num.append(log_batch_step + previous_batch)
                    cost_batch.append(l)
            valAcc = self.getValAccuracy(sess=sess)
            testAcc = self.getTestAccuracy(sess=sess)
            val_acc_epoch.append(valAcc)
            test_acc_epoch.append(testAcc)
            print("epoch {}, valAcc={},testAcc={}".format(epoch_i,valAcc,testAcc))
        t = strftime("%Y-%m-%d-%H-%M-%S", localtime())
        result_dir = "./"+t
        os.mkdir(result_dir)
        history = {"val_acc_epoch":val_acc_epoch,"test_acc_epoch":test_acc_epoch,"cost_batch":cost_batch}
        with open(result_dir+'history.pickle', 'wb') as f:
            pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)
        save_path = saver.save(sess, result_dir+"/model-"+t)
        print("Model saved in file: %s" % save_path)

    def debug_tf(self):
        gen = self.generate_from_directory(self.X_train_paths, self.y_train, batch_size=4)
        X, Y = next(gen)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        temp = self.model.Y_true * tf.log(tf.clip_by_value(self.model.Y_pred, 1e-10, 1))
        Y_true, Y_pred, cross_entropy, temp= sess.run([self.model.Y_true,self.model.Y_pred,self.model.cross_entropy,temp],\
                                                feed_dict={self.model.X:X, self.model.Y_true:Y})
        pass