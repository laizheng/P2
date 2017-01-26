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
    def __init__(self, use_gray=False, use_jitter=False):
        self.use_gray = use_gray
        self.use_jitter = use_jitter
        self.img_rows, self.img_cols = 32, 32
        if self.use_gray:
            self.input_shape = (self.img_rows, self.img_cols)
        else:
            self.input_shape = (self.img_rows, self.img_cols, 3)
        self.nb_classes = 43
        training_file = "./train.p"
        testing_file = "./test.p"
        with open(training_file, mode='rb') as f:
            train = pickle.load(f)
        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)
        self.X_train, self.y_train = train['features'], train['labels']
        self.X_test, self.y_test = test['features'], test['labels']
        self.train_ids = []
        self.val_ids = []
        self.test_ids = np.array(list(range(len(self.y_test))))
        self.samplesPerTrack = {}
        self.labelDict = self.readLabelFromFile()
        self.encoder = LabelBinarizer()
        self.encoder.fit(np.unique(self.y_train))
        self.cat()
        self.split()
        self.model = Model(input_shape=self.input_shape)

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
        tracks = np.unique(self.y_train)  # a list of unique tracks
        self.samplesPerTrack = {} # store the id to image/y
        for track in tracks:
            self.samplesPerTrack[track] = []
            select = np.reshape(np.argwhere(self.y_train==track),(len(np.argwhere(self.y_train==track)),))
            self.samplesPerTrack[track].extend(select)
        #if self.use_jitter:
        #    for track in tracks:
        #        self.samplesPerTrack[track].extend(list(self.df_jitter[self.df_jitter['y'] == track]['path']))
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
            ids = self.samplesPerTrack[track]
            ids_train, ids_val = train_test_split(ids, test_size=0.2, random_state=42)
            self.train_ids.extend(ids_train)
            self.val_ids.extend(ids_val)
        self.train_ids = np.array(self.train_ids)
        self.val_ids = np.array(self.val_ids)
        print("Train #={}".format(len(self.train_ids)))
        print("Val #={}".format(len(self.val_ids)))
        print("Splitting completes.")

    def generate_from_ids(self, X, y, ids, batch_size=32, break_if_finish=False):
        assert len(X) == len(y)
        Y = self.encoder.transform(y)
        start_index = 0
        while 1:
            if start_index + batch_size <= len(ids):
                X_ret = []
                for j in range(batch_size):
                    x = X[ids[start_index+j]]
                    x = self.imgPreprocess(x)
                    X_ret.append(x)
                X_ret = np.array(X_ret)
                yield (X_ret, Y[ids[start_index: start_index + batch_size], :])
                start_index = start_index + batch_size
            else:
                X_ret = []
                for j in range(len(ids) - start_index):
                    x = X[ids[start_index+j]]
                    x = self.imgPreprocess(x)
                    X_ret.append(x)
                X_ret = np.array(X_ret)
                yield (X_ret, Y[ids[start_index: len(ids)], :])
                if break_if_finish:
                    return
                start_index = 0

    def getValAccuracy(self,sess):
        print("computing Val Acc..."),
        acc = []
        for X, Y in tqdm(self.generate_from_ids(self.X_train,self.y_train,self.val_ids,batch_size=32,break_if_finish=True), \
                         desc='Val Acc'):
            acc.append(sess.run(self.model.accuracy,feed_dict={self.model.X:X, self.model.Y_true:Y}))
        return np.mean(acc)

    def getTestAccuracy(self,sess):
        print("computing Test Acc..."),
        acc = []
        for X, Y in tqdm(self.generate_from_ids(self.X_train,self.y_train,self.test_ids,batch_size=32,break_if_finish=True), \
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
        print("Training Begin...")
        for epoch_i in range(epochs):
            self.train_ids = shuffle(self.train_ids)
            gen = self.generate_from_ids(self.X_train, self.y_train, self.train_ids, batch_size=batch_size)
            batch_count = int(len(self.train_ids)/batch_size) + 1
            #batches_pbar = range(batch_count)
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
            val_acc_epoch.append(valAcc)
            print("epoch {}, valAcc={}".format(epoch_i,valAcc))
        t = strftime("%Y-%m-%d-%H-%M-%S", localtime())
        result_dir = "./"+t
        os.mkdir(result_dir)
        history = {"val_acc_epoch":val_acc_epoch,"cost_batch":cost_batch}
        with open(result_dir+'history.pickle', 'wb') as f:
            pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)
        save_path = saver.save(sess, result_dir+"/model-"+t)
        print("Model saved in file: %s" % save_path)

    def debug_tf(self):
        gen = self.generate_from_ids(self.X_train, self.y_train, self.train_ids, batch_size=4)
        X, Y = next(gen)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        temp = self.model.Y_true * tf.log(tf.clip_by_value(self.model.Y_pred, 1e-10, 1))
        Y_true, Y_pred, cross_entropy, fc2, numeric_stable, wconv1 = sess.run(
            [self.model.Y_true, self.model.Y_pred, self.model.cross_entropy, self.model.layer_fc2,
             self.model.numeric_stable_out, self.model.weights_conv1],
            feed_dict={self.model.X: X[10:11], self.model.Y_true: Y[10:11]})
        pass