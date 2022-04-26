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

# train processing

def set_para():
    global file_name
    global model_record_path
    global file_record_path
    global method_name
    global model_type
    global mirror_type
    global kernelpca_or_not
    global pca_or_not
    global num_of_components

    global scaler_name
    global pca_name
    global kernelpca_name
    global model_name

    argv = sys.argv[1:]
    for each in argv:
        para = each.split('=')
        if para[0] == 'file_name':
            file_name = para[1]
        if para[0] == 'model_record_path':
            model_record_path = para[1]
        if para[0] == 'file_record_path':
            file_record_path = para[1]
        if para[0] == 'method_name':
            method_name = para[1]
        if para[0] == 'model_type':
            model_type = para[1].upper()
        if para[0] == 'mirror_type':
            mirror_type = para[1]
        if para[0] == 'kernelpca':
            if para[1] == 'True':
                kernelpca_or_not = True
            else:
                kernelpca_or_not = False
        if para[0] == 'pca':
            if para[1] == 'True':
                pca_or_not = True
            else:
                pca_or_not = False
        if para[0] == 'num_of_components':
            num_of_components = int(para[1])

        if para[0] == 'scaler_name':
            scaler_name = para[1]
        if para[0] == 'pca_name':
            pca_name = para[1]
        if para[0] == 'kernelpca_name':
            kernelpca_name = para[1]
        if para[0] == 'model_name':
            model_name = para[1]

    if kernelpca_or_not and pca_or_not:
        pca_or_not = True
        kernelpca_or_not = False

# -------------------------------------parameters----------------------------------------
file_name = 'GData_train.csv'
model_record_path = '../1_year_result/model/'
file_record_path = '../1_year_result/record/'
method_name = "smote"
# model_type = 'LR'
model_type = 'SVC'
# model_type = 'DT'
# mirror_type = "mirror"
mirror_type = "not_mirror"
kernelpca_or_not = False
pca_or_not = False
num_of_components = 19

scaler_name = 'scaler.m'
pca_name = 'pca.m'
kernelpca_name = ''
model_name = 'model.m'
positive_value = 1
negative_value = -1
threshold_value = 0
winner_number = 3


train_times = 50000

# ----------------------------------set parameters---------------------------------------
set_para()

# ----------------------------------start processing-------------------------------------
print(file_name)

# file_number = re.findall(r"\d+", file_name)[-1]
scaler_name = model_record_path + method_name + '_' + scaler_name
if pca_or_not:
    pca_name = model_record_path + method_name + '_' + pca_name
if kernelpca_or_not:
    kernelpca_name = model_record_path  + method_name + '_' + kernelpca_name
model_name = model_record_path + method_name + '_' + model_name

# data input
data, label = handle_data.loadTrainData(file_name)


data = data.astype(np.float64)
train_label = label.astype(np.int)

start = clock()
new_data = handle_data.standarize_PCA_data(data, pca_or_not, kernelpca_or_not, num_of_components, scaler_name, pca_name, kernelpca_name)

train_data = new_data
batch_size = 50


positive_data, negative_data = handle_data.divide_data(train_data, train_label)

# create LogisticRegression model


single_input_size = train_data.shape[1]
transformed_input_size = 2 * single_input_size


num_class = 1

# x = tf.placeholder(tf.float32, [None, single_input_size])
x = tf.placeholder(tf.float32, [None, 2, single_input_size])
y_true = tf.placeholder(tf.float32, [None, 2, num_class])
y_transformed_true = tf.placeholder(tf.float32, [None, num_class])


# # one hidden layer ------------------------------------------------
hidden1 = tf.layers.dense(inputs=x, units=3*single_input_size, use_bias=True, activation=tf.nn.relu)
# y_pred = tf.layers.dense(inputs=hidden1, units=4, activation=tf.nn.sigmoid)
y_pred = tf.layers.dense(inputs=hidden1, units=num_class, activation=tf.nn.sigmoid)

y_transformed = tf.math.sigmoid(10 * (y_pred[:,0,0] -  y_pred[:,1,0]))
y_transformed = tf.reshape(y_transformed, shape=(-1,1))


# loss_1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[:,0,0], logits=y_pred[:,0,0])
# loss_2 = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[:,1,0], logits=y_pred[:,1,0])




loss_1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

loss_2 = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_transformed_true, logits=y_transformed)

# loss = loss_1 + loss_2

loss = loss_1

# print(loss)

cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(cost)


# two hidden layer ------------------------------------------------
# hidden1 = tf.layers.dense(inputs=x, units=2*transformed_input_size, use_bias=True, activation=tf.nn.sigmoid)
# hidden2 = tf.layers.dense(inputs=hidden1, units=2*transformed_input_size, use_bias=True, activation=tf.nn.sigmoid)
# # y_pred = tf.layers.dense(inputs=hidden1, units=4, activation=tf.nn.sigmoid)
# y_pred = tf.layers.dense(inputs=hidden2, units=4, activation=tf.nn.sigmoid)

# three hidden layer ------------------------------------------------
# hidden1 = tf.layers.dense(inputs=x, units=2*transformed_input_size, use_bias=True, activation=tf.nn.relu)
# hidden2 = tf.layers.dense(inputs=hidden1, units=2*transformed_input_size, use_bias=True, activation=tf.nn.relu)
# hidden3 = tf.layers.dense(inputs=hidden2, units=transformed_input_size, use_bias=True, activation=tf.nn.relu)
# # y_pred = tf.layers.dense(inputs=hidden1, units=4, activation=tf.nn.sigmoid)
# y_pred = tf.layers.dense(inputs=hidden3, units=4, activation=tf.nn.sigmoid)


# one hidden layer ------------------------------------------------
# hidden1 = tf.layers.dense(inputs=x, units=2*single_input_size, use_bias=True, activation=tf.nn.relu)
# # y_pred = tf.layers.dense(inputs=hidden1, units=4, activation=tf.nn.sigmoid)
# y_pred = tf.layers.dense(inputs=hidden1, units=num_class, activation=tf.nn.sigmoid)


# 4 hidden layer --------------------------------------------------
# hidden1 = tf.layers.dense(inputs=x, units=2*single_input_size, use_bias=True, activation=tf.nn.relu)
# hidden2 = tf.layers.dense(inputs=hidden1, units=2*transformed_input_size, use_bias=True, activation=tf.nn.relu)
# hidden3 = tf.layers.dense(inputs=hidden2, units=transformed_input_size, use_bias=True, activation=tf.nn.relu)
# hidden4 = tf.layers.dense(inputs=hidden3, units=single_input_size, use_bias=True, activation=tf.nn.relu)
# # y_pred = tf.layers.dense(inputs=hidden1, units=4, activation=tf.nn.sigmoid)
# y_pred = tf.layers.dense(inputs=hidden4, units=num_class, activation=tf.nn.sigmoid)




# 5 hidden layers -------------------------------------------------
# hidden1 = tf.layers.dense(inputs=x, units=2*transformed_input_size, use_bias=True, activation=tf.nn.relu)
# hidden2 = tf.layers.dense(inputs=hidden1, units=2*transformed_input_size, use_bias=True, activation=tf.nn.relu)
# hidden3 = tf.layers.dense(inputs=hidden2, units=transformed_input_size, use_bias=True, activation=tf.nn.relu)
# hidden4 = tf.layers.dense(inputs=hidden3, units=single_input_size, use_bias=True, activation=tf.nn.relu)
# hidden5 = tf.layers.dense(inputs=hidden4, units=2*num_class, use_bias=True, activation=tf.nn.relu)
# # y_pred = tf.layers.dense(inputs=hidden1, units=4, activation=tf.nn.sigmoid)
# y_pred = tf.layers.dense(inputs=hidden4, units=num_class, activation=tf.nn.sigmoid)

# 8 hidden layers -----------------------------------------------------------
# hidden1 = tf.layers.dense(inputs=x, units=4*transformed_input_size, use_bias=True, activation=tf.nn.relu)
# hidden2 = tf.layers.dense(inputs=hidden1, units=3*transformed_input_size, use_bias=True, activation=tf.nn.relu)
# hidden3 = tf.layers.dense(inputs=hidden2, units=2*transformed_input_size, use_bias=True, activation=tf.nn.relu)
# hidden4 = tf.layers.dense(inputs=hidden3, units=transformed_input_size, use_bias=True, activation=tf.nn.relu)
# hidden5 = tf.layers.dense(inputs=hidden4, units=single_input_size, use_bias=True, activation=tf.nn.relu)
# hidden6 = tf.layers.dense(inputs=hidden5, units=4*num_class, use_bias=True, activation=tf.nn.relu)
# hidden7 = tf.layers.dense(inputs=hidden6, units=2*num_class, use_bias=True, activation=tf.nn.relu)
# # y_pred = tf.layers.dense(inputs=hidden1, units=4, activation=tf.nn.sigmoid)
# y_pred = tf.layers.dense(inputs=hidden7, units=num_class, activation=tf.nn.sigmoid)




tf.add_to_collection('x', x)
tf.add_to_collection('y_true', y_true)
tf.add_to_collection('y_pred', y_pred)
tf.add_to_collection('cost', cost)
tf.add_to_collection('optimizer', optimizer)
saver = tf.train.Saver()


# create session and run the model
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.46)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())


for i in range(train_times):
    # train_data, train_label = handle_data.generate_batch_data(positive_data, negative_data, batch_size)
    train_data, train_label, transformed_label = handle_data.next_batch(positive_data, negative_data)
    train_data = np.array(train_data).reshape((-1,2,single_input_size))
    train_label = np.array(train_label).reshape((-1,2,1))
    transformed_label = np.array(transformed_label).reshape((-1,1))
    feed_dict_train = {
        x                   : train_data,
        y_true              : train_label,
        y_transformed_true  : transformed_label

    }

    cost_val, true_label, pred_label, opt_obj = sess.run( [cost, y_true, y_pred,
        optimizer], feed_dict=feed_dict_train )
    if (i % 1000) == 0 :
        print('epoch: {0} cost = {1}'.format(i,cost_val))

finish = clock()
saver.save(sess, model_name)

running_time = finish-start
print('running time is {0}'.format(running_time))
