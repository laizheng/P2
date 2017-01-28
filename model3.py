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

class Model():
    def __init__(self,input_shape):
        self.img_size = 32
        self.img_shape = (self.img_size, self.img_size)
        self.num_classes = 43
        self.input_shape = input_shape
        self.getModel()

    def new_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    def new_biases(sefl, length):
        return tf.Variable(tf.constant(0.05, shape=[length], dtype=tf.float32))

    def new_conv_layer(self,
                       input,  # The previous layer.
                       num_input_channels,  # Num. channels in prev. layer.
                       filter_size,  # Width and height of each filter.
                       num_filters,  # Number of filters.
                       strides,  # sub-sample size
                       padding,  # "SAME" OR "VALID"
                       use_pooling=True):  # Use 2x2 max-pooling.

        shape = [filter_size, filter_size, num_input_channels, num_filters]
        weights = self.new_weights(shape=shape)
        biases = self.new_biases(length=num_filters)
        # Create the TensorFlow operation for convolution.
        # Note the strides are set to 1 in all dimensions.
        # The first and last stride must always be 1,
        # because the first is for the image-number and
        # the last is for the input-channel.
        # But e.g. strides=[1, 2, 2, 1] would mean that the filter
        # is moved 2 pixels across the x- and y-axis of the image.
        # The padding is set to 'SAME' which means the input image
        # is padded with zeroes so the size of the output is the same.
        layer = tf.nn.conv2d(input=input,
                             filter=weights,
                             strides=[1, strides, strides, 1],
                             padding=padding)
        layer += biases
        if use_pooling:
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')
        layer = tf.nn.relu(layer)
        return layer, weights

    def flatten_layer(self, layer):
        layer_shape = layer.get_shape()
        # The shape of the input layer is assumed to be:
        # layer_shape == [num_images, img_height, img_width, num_channels]

        # The number of features is: img_height * img_width * num_channels
        # We can use a function from TensorFlow to calculate this.
        num_features = layer_shape[1:4].num_elements()

        # Reshape the layer to [num_images, num_features].
        # Note that we just set the size of the second dimension
        # to num_features and the size of the first dimension to -1
        # which means the size in that dimension is calculated
        # so the total size of the tensor is unchanged from the reshaping.
        layer_flat = tf.reshape(layer, [-1, num_features])

        # The shape of the flattened layer is now:
        # [num_images, img_height * img_width * num_channels]
        return layer_flat, num_features

    def new_fc_layer(self,
                     input,  # The previous layer.
                     num_inputs,  # Num. inputs from prev. layer.
                     num_outputs,  # Num. outputs.
                     use_relu=True):  # Use Rectified Linear Unit (ReLU)?

        # Create new weights and biases.
        weights = self.new_weights(shape=[num_inputs, num_outputs])
        biases = self.new_biases(length=num_outputs)

        # Calculate the layer as the matrix multiplication of
        # the input and weights, and then add the bias-values.
        layer = tf.matmul(input, weights) + biases

        # Use ReLU?
        if use_relu:
            layer = tf.nn.relu(layer)

        return layer, weights

    def getModel(self):
        self.X = tf.placeholder(tf.float32, shape= [None] + list(self.input_shape), name='X')
        if len(self.input_shape) == 2:
            self.X_reshape = tf.reshape(self.X, [-1, self.img_size, self.img_size, 1])
        else:
            self.X_reshape = self.X
        self.Y_true = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='Y')
        self.y_true = tf.argmax(self.Y_true, dimension=1)
        self.layer_conv1, self.weights_conv1 = \
            self.new_conv_layer(input=self.X_reshape,
                           num_input_channels=1 if len(self.input_shape) == 2 else 3,
                           filter_size=3,
                           num_filters=16,
                           strides=1,
                           padding='SAME',
                           use_pooling=True)
        self.layer_conv2, self.weights_conv2 = \
            self.new_conv_layer(input=self.layer_conv1,
                           num_input_channels=16,
                           filter_size=3,
                           num_filters=24,
                           strides=1,
                           padding='SAME',
                           use_pooling=False)
        #self.dropout0 = tf.nn.dropout(self.layer_conv2, keep_prob=0.4)
        self.layer_conv3, self.weights_conv3 = \
            self.new_conv_layer(input=self.layer_conv2,
                           num_input_channels=24,
                           filter_size=3,
                           num_filters=36,
                           strides=1,
                           padding='SAME',
                           use_pooling=False)
        self.layer_conv4, self.weights_conv4 = \
            self.new_conv_layer(input=self.layer_conv3,
                           num_input_channels=36,
                           filter_size=3,
                           num_filters=48,
                           strides=2,
                           padding='SAME',
                           use_pooling=False)
        self.dropout1 = tf.nn.dropout(self.layer_conv4, keep_prob=0.7)
        self.layer_conv5, self.weights_conv5 = \
            self.new_conv_layer(input=self.dropout1,
                                num_input_channels=48,
                                filter_size=3,
                                num_filters=60,
                                strides=2,
                                padding='SAME',
                                use_pooling=False)
        self.layer_conv6, self.weights_conv6 = \
            self.new_conv_layer(input=self.layer_conv5,
                                num_input_channels=60,
                                filter_size=3,
                                num_filters=72,
                                strides=2,
                                padding='SAME',
                                use_pooling=False)
        self.dropout2 = tf.nn.dropout(self.layer_conv6, keep_prob=0.7)
        self.layer_flat, self.num_flat_features = self.flatten_layer(self.dropout2)
        self.layer_fc1, self.weights_fc1 = self.new_fc_layer(input=self.layer_flat,
                                 num_inputs=self.num_flat_features,
                                 num_outputs=128,
                                 use_relu=True)
        self.dropout3 = tf.nn.dropout(self.layer_fc1, keep_prob=0.7)
        self.layer_fc2, self.weights_fc2 = self.new_fc_layer(input=self.dropout3,
                                 num_inputs=128,
                                 num_outputs=self.num_classes,
                                 use_relu=False)
        self.numeric_stable_out = self.layer_fc2-tf.reshape(tf.reduce_max(self.layer_fc2,axis=1), [-1,1])
        self.Y_pred = tf.nn.softmax(self.numeric_stable_out)
        self.y_pred = tf.argmax(self.Y_pred, dimension=1)
        self.cross_entropy = -tf.reduce_sum(self.Y_true * tf.log(tf.clip_by_value(self.Y_pred,1e-10,1)), reduction_indices=1)
        self.cost = tf.reduce_mean(self.cross_entropy)
        self.correct_prediction = tf.equal(self.y_pred, self.y_true)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.cost)