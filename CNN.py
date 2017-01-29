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
from model3 import Model
from tqdm import tqdm
from sklearn.utils import shuffle
from time import localtime, strftime
from collections import Counter
from sklearn.metrics import confusion_matrix
#from tensorflow.python import debug as tf_debug

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
        self.X_train_orig_paths = list(self.df_train['path'])
        self.y_train_orig = list(self.df_train['y'])
        self.X_test_paths = list(self.df_test['path'])
        self.y_test = list(self.df_test['y'])
        self.encoder = LabelBinarizer()
        self.encoder.fit(list(self.df_train["y"]))
        self.cat()
        self.split()
        self.model = Model(input_shape=self.input_shape)
        t = strftime("%Y-%m-%d-%H-%M-%S", localtime())
        self.log_dir = "./" + t + '/'

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
                    x = cv2.imread(X_paths[start_index+j])
                    x = self.imgPreprocess(x)
                    X.append(x)
                X = np.array(X)
                yield (X, Y[start_index: start_index + batch_size,:])
                start_index = start_index + batch_size
            else:
                X = []
                for j in range(len(X_paths) - start_index):
                    x = cv2.imread(X_paths[start_index+j])
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
        print("valAcc={}".format(np.mean(acc)))
        return np.mean(acc)

    def getTestAccuracy(self,sess):
        print("computing Test Acc..."),
        acc = []
        y_true =[]
        y_pred = []
        for X, Y in tqdm(self.generate_from_directory(self.X_test_paths,self.y_test,batch_size=32,break_if_finish=True), \
                         desc='Test Acc'):
            acc.append(sess.run(self.model.accuracy,feed_dict={self.model.X:X, self.model.Y_true:Y}))
            y_true.extend(sess.run(self.model.y_true,feed_dict={self.model.X:X, self.model.Y_true:Y}))
            y_pred.extend(sess.run(self.model.y_pred, feed_dict={self.model.X: X, self.model.Y_true: Y}))
        df = pd.DataFrame({"path":self.X_test_paths, "y_true":y_true, "y_pred":y_pred})
        print("testAcc={}".format(np.mean(acc)))
        return np.mean(acc), df

    def getTrainAccuracy(self,sess):
        print("computing Train Acc..."),
        acc = []
        for X, Y in tqdm(self.generate_from_directory(self.X_train_paths,self.y_train,batch_size=32,break_if_finish=True), \
                         desc='Train Acc'):
            acc.append(sess.run(self.model.accuracy,feed_dict={self.model.X:X, self.model.Y_true:Y}))
        print("trainAcc={}".format(np.mean(acc)))
        return np.mean(acc)

    def getOrigTrainAccuracy(self,sess):
        print("computing Train Acc(entire training set, no spliting)..."),
        acc = []
        y_true = []
        y_pred = []
        for X, Y in tqdm(self.generate_from_directory(self.X_train_orig_paths,self.y_train_orig,batch_size=32,break_if_finish=True), \
                         desc='Train Acc'):
            acc.append(sess.run(self.model.accuracy,feed_dict={self.model.X:X, self.model.Y_true:Y}))
            y_true.extend(sess.run(self.model.y_true, feed_dict={self.model.X: X, self.model.Y_true: Y}))
            y_pred.extend(sess.run(self.model.y_pred, feed_dict={self.model.X: X, self.model.Y_true: Y}))
        df = pd.DataFrame({"path": self.X_train_orig_paths, "y_true": y_true, "y_pred": y_pred})
        print("OrigTrainAcc={}".format(np.mean(acc)))
        return np.mean(acc), df

    def train(self,epochs,batch_size):
        os.mkdir(self.log_dir)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        file_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
        log_batch_step = 10
        batch_num = []
        cost_batch = []
        val_acc_epoch = []
        train_acc_epoch = []
        print("Training Begin...")
        for epoch_i in range(epochs):
            self.X_train_paths, self.y_train = shuffle(self.X_train_paths,self.y_train)
            gen = self.generate_from_directory(self.X_train_paths, self.y_train, batch_size=batch_size)
            batch_count = int(len(self.X_train_paths)/batch_size) + 1
            batches_pbar = tqdm(range(batch_count), \
                                desc='Epoch {:>2}/{}'.format(epoch_i + 1, epochs))
            for batch_cnt in batches_pbar:
                X, Y = next(gen)
                # Run optimizer and get loss
                _, l = sess.run(
                    [self.model.optimizer, self.model.cost],
                    feed_dict={self.model.X:X, self.model.Y_true:Y})
                if not batch_cnt % log_batch_step:
                    previous_batch = batch_num[-1] if batch_num else 0
                    batch_num.append(log_batch_step + previous_batch)
                    cost_batch.append(l)
                """
                wconv1, wconv2, wconv3, wconv4, wconvfc1, wconvfc2, Y_pred = \
                    sess.run([self.model.weights_conv1, self.model.weights_conv2, self.model.weights_conv3, \
                              self.model.weights_conv4, self.model.weights_fc1, self.model.weights_fc2,
                              self.model.Y_pred],
                             feed_dict={self.model.X: X, self.model.Y_true: Y})
                if np.isnan(np.sum(wconv1)) or np.isnan(np.sum(wconv2)) or np.isnan(np.sum(wconv3)) or np.isnan(
                        np.sum(wconv4)) \
                        or np.isnan(np.sum(wconvfc1)) or np.isnan(np.sum(wconvfc2)) or np.isnan(np.sum(Y_pred)):
                    raise ValueError("nan detected")
                """
            trainAcc = self.getTrainAccuracy(sess=sess)
            train_acc_epoch.append(trainAcc)
            valAcc = self.getValAccuracy(sess=sess)
            val_acc_epoch.append(valAcc)
            history = {"val_acc_epoch":val_acc_epoch,"val_acc_epoch":val_acc_epoch,"cost_batch":cost_batch}
            with open(self.log_dir+'/history.pickle', 'wb') as f:
                pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)
            save_path = saver.save(sess, self.log_dir+"/model.ckt")
            print("Model saved in file: %s" % save_path)
        self.getTrainAccuracy(sess=sess)
        self.getTestAccuracy(sess=sess)

    def trainNoSplit(self,epochs,batch_size):
        os.mkdir(self.log_dir)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        file_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
        log_batch_step = 10
        batch_num = []
        cost_batch = []
        test_acc_epoch = []
        train_acc_epoch = []
        print("Training Begin...")
        for epoch_i in range(epochs):
            self.X_train_orig_paths, self.y_train_orig = shuffle(self.X_train_orig_paths,self.y_train_orig)
            gen = self.generate_from_directory(self.X_train_orig_paths, self.y_train_orig, batch_size=batch_size)
            batch_count = int(len(self.X_train_orig_paths)/batch_size) + 1
            batches_pbar = tqdm(range(batch_count), \
                                desc='Epoch {:>2}/{}'.format(epoch_i + 1, epochs))
            for batch_cnt in batches_pbar:
                X, Y = next(gen)
                # Run optimizer and get loss
                _, l = sess.run(
                    [self.model.optimizer, self.model.cost],
                    feed_dict={self.model.X:X, self.model.Y_true:Y})
                if not batch_cnt % log_batch_step:
                    previous_batch = batch_num[-1] if batch_num else 0
                    batch_num.append(log_batch_step + previous_batch)
                    cost_batch.append(l)
            testAcc = self.getTestAccuracy(sess=sess)
            test_acc_epoch.append(testAcc)
            trainAcc = self.getOrigTrainAccuracy(sess=sess)
            train_acc_epoch.append(trainAcc)
            history = {"test_acc_epoch":test_acc_epoch,"train_acc_epoch":train_acc_epoch, "cost_batch":cost_batch}
            with open(self.log_dir+'/history.pickle', 'wb') as f:
                pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)
            save_path = saver.save(sess, self.log_dir+"/model.ckt")
            print("Model saved in file: %s" % save_path)
        self.getTrainAccuracy(sess=sess)
        self.getTestAccuracy(sess=sess)

    def plot_five_images_per_track(self,df): # to be used in iPython
        for id in np.unique(df['y']):
            desc = self.labelDict[str(id)]
            print(str(id) + ":" + desc)
            pick = np.random.choice(df[df['y']==id]['path'],5)
            fig, axes = plt.subplots(1, 5)
            for j, ax in enumerate(axes.flat):
                img = cv2.imread(pick[j])
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
            plt.tight_layout()
            plt.show()

    def plot_error_examples(self,df): # to be used in iPython
        df_error = df.iloc[np.where(df['y_pred']!=df['y_true'])]
        for true_id in np.unique(df_error['y_true']):
            pick = np.random.choice(df_error[df_error['y_true']==true_id].index,3)
            title = "true:" + str(true_id) + "\n"
            fig, axes = plt.subplots(1, 3)
            for j, ax in enumerate(axes.flat):
                img = cv2.imread(df.iloc[pick[j]]['path'])
                pred_id = df.iloc[pick[j]]['y_pred']
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(title + "pred:" + str(pred_id))
            plt.tight_layout()
            plt.show()

    def debug_tf(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        gen = self.generate_from_directory(self.X_train_paths, self.y_train, batch_size=4)
        X, Y = next(gen)
        wconv1, wconv2, wconv3, wconv4, wconvfc1, wconvfc2, numeric_stable, Y_pred = \
            sess.run([self.model.weights_conv1, self.model.weights_conv2, self.model.weights_conv3, \
                      self.model.weights_conv4, self.model.weights_fc1, self.model.weights_fc2, \
                      self.model.numeric_stable_out, self.model.Y_pred],
                     feed_dict={self.model.X: X, self.model.Y_true: Y})
        pass

    def print_full(self, x):
        pd.set_option('display.max_columns', 50)
        print(x)
        pd.reset_option('display.max_columns')

    def test_saved_model(self,model_path, show_confusion = True):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        train_acc, df_train_trained = self.getOrigTrainAccuracy(sess)
        test_acc, df_test_trained = self.getTestAccuracy(sess)

        if show_confusion:
            cm = confusion_matrix(y_true=df_train_trained['y_true'], y_pred=df_train_trained['y_pred'])
            print("training set confusion matrix")
            self.print_full(pd.DataFrame(cm))
            plt.figure(num=1, figsize=(12, 12))
            plt.matshow(cm, fignum=1)
            plt.colorbar()
            tick_marks = np.arange(self.nb_classes)
            plt.xticks(tick_marks, range(self.nb_classes))
            plt.yticks(tick_marks, range(self.nb_classes))
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title("Error in training set")
            plt.show()

            cm = confusion_matrix(y_true=df_test_trained['y_true'], y_pred=df_test_trained['y_pred'])
            print("test set confusion matrix")
            self.print_full(pd.DataFrame(cm))
            plt.figure(num=1, figsize=(12, 12))
            plt.matshow(cm, fignum=1)
            plt.colorbar()
            tick_marks = np.arange(self.nb_classes)
            plt.xticks(tick_marks, range(self.nb_classes))
            plt.yticks(tick_marks, range(self.nb_classes))
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title("Error in test set")
            plt.show()

        return df_train_trained, df_test_trained

    def predict(self,img, model_path):
        processed_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        processed_img = cv2.resize(processed_img,(self.img_cols,self.img_rows))
        processed_img = self.imgPreprocess(processed_img)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            top_k = tf.nn.top_k(self.model.Y_pred, k=5, sorted=True, name=None)
            pred = sess.run(top_k, feed_dict={self.model.X:[processed_img]})
            return pred.values[0], pred.indices[0]