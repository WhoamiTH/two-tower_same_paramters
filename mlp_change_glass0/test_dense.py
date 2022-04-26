# train processing head
import sys
import sklearn.svm as sksvm
import sklearn.linear_model as sklin
import sklearn.tree as sktree
from sklearn.externals import joblib
from time import clock
import handle_data
import predict_test
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

single_input_size = 3
transformed_input_size = 2 * single_input_size


num_class = 1

x1 = tf.constant([[1],[2],[3]], dtype=tf.float32)
x2 = tf.constant([[1],[2],[3]], dtype=tf.float32)

# one hidden layer ------------------------------------------------

def mymodel(x):
    hidden1 = tf.layers.dense(inputs=x1, units=3*single_input_size, use_bias=True, activation=tf.nn.relu, name='hidden1')
    # y_pred = tf.layers.dense(inputs=hidden1, units=4, activation=tf.nn.sigmoid)
    y_pred_1 = tf.layers.dense(inputs=hidden1, units=num_class, activation=tf.nn.sigmoid, name='y_pred')
    return y_pred_1

y1 = mymodel(x1)
y2 = mymodel(x2)

# hidden2 = tf.layers.dense(inputs=x2, units=3*single_input_size, use_bias=True, activation=tf.nn.relu, name='hidden1', reuse=True)
# y_pred_2 = tf.layers.dense(inputs=hidden2, units=num_class, activation=tf.nn.sigmoid,  name='y_pred', reuse=True)

# x = tf.ones((1, 3))
# hidden1 = tf.layers.dense(x1, units=3*single_input_size, use_bias=True, activation=tf.nn.sigmoid, name='hidden1')
# y1 = tf.layers.dense(inputs=hidden1, units=3, use_bias=True, activation=tf.nn.sigmoid, name='y1')

# hidden2 = tf.layers.dense(x2, units=3*single_input_size, use_bias=True, activation=tf.nn.sigmoid, name='hidden1', reuse=True)
# y2 = tf.layers.dense(inputs=hidden2, units=3, use_bias=True, activation=tf.nn.sigmoid, name='y1', reuse=True)
# y1 and y2 will evaluate to the same values
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(y1))
print(sess.run(y2)) 



# # # create session and run the model
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.46)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# sess.run(tf.global_variables_initializer())

# # with tf.Session() as sess:
# y1 = sess.run(y_pred_1)
# y2 = sess.run(y_pred_2)
# print(y1)
# print(y2)