import sys
from sklearn.externals import joblib
from time import clock
import handle_data
import predict_test
import numpy as np
import csv
import re
import pandas as pd
import tensorflow as tf
import sklearn.metrics as skmet


import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def set_para():
    global train_file_name
    global test_file_name
    global model_record_path
    global file_record_path
    global method_name

    global scaler_name
    global kernelpca_name
    global pca_name
    global model_name
    global record_name

    argv = sys.argv[1:]
    for each in argv:
        para = each.split('=')
        if para[0] == 'train_file_name':
            train_file_name = para[1]
        if para[0] == 'test_file_name':
            test_file_name = para[1]
        if para[0] == 'model_record_path':
            model_record_path = para[1]
        if para[0] == 'file_record_path':
            file_record_path = para[1]
        if para[0] == 'method_name':
            method_name = para[1]

        if para[0] == 'scaler_name':
            scaler_name = para[1]
        if para[0] == 'kernelpca_name':
            kernelpca_name = para[1]
        if para[0] == 'pca_name':
            pca_name = para[1]
        if para[0] == 'model_name':
            model_name = para[1]
        if para[0] == 'record_name':
            record_name = para[1]

    if kernelpca_name and pca_name:
        kernelpca_name = ''


# -------------------------------------global parameters-------------------------------
train_file_name = 'NBA_train.csv'
# train file is the file which contained the reference data
test_file_name = 'NBA_test.csv'
model_record_path = '../1_year_result/model/'
file_record_path = '../1_year_result/record/'
method_name = "smote"

scaler_name = 'scaler.m'
kernelpca_name = ''
pca_name = ''
model_name = 'model.m'
threshold_value = 0
record_name = 'result.csv'

winner_number = 3
# test time is the reference number of test
test_time = 10

# ----------------------------------set parameters--------------------------------------
set_para()

# ----------------------------------start processing------------------------------------
print(train_file_name)
print(test_file_name)

scaler_name = model_record_path + method_name + '_' + scaler_name
if pca_name != '':
    pca_name = model_record_path + method_name + '_' + pca_name
if kernelpca_name != '':
    kernelpca_name = model_record_path  + method_name + '_' + kernelpca_name
model_name = model_record_path  + method_name + '_' + model_name
record_name = file_record_path + method_name + '_' + record_name

print(model_name)

# ------------- load train data and find reference data --------------------------------
# train_data, train_label = handle_data.loadTrainData(train_file_name)

# group_index_list = handle_data.group(train_data)

# train_data = train_data.values
# train_data = train_data.astype(np.float64)

# train_label = train_label.astype(np.int)
# # as there are some errors in the labels, for example, only 2 lables are left, one is 2 another one is 10
# # the labels have to be transformed into a safe mode, for instance, for the case informed above, 2 will be transformed to 1, and 10 will be transformed to 2
# train_label = predict_test.transform_labels(train_label, group_index_list, winner_number)

# train_data = handle_data.transform_data_by_standarize_pca(train_data, scaler_name, pca_name, kernelpca_name)

# positive_data, negative_data = handle_data.divide_data(train_data, train_label)
# negative data is the reference data

# ---------------------- load test data ------------------------------------------------
file_data, test_data = handle_data.loadTestData(test_file_name)
test_data, test_label = handle_data.loadTrainData(test_file_name)
test_data = test_data.values
test_data = test_data.astype(np.float64)

start = clock()
test_data = handle_data.transform_data_by_standarize_pca(test_data, scaler_name, pca_name, kernelpca_name)


# transform_test_data
# test_data = list(test_data)
# transformed_test_data = []
# for i in test_data:
#     transformed_test_data.append([list(i), list(i)])
# test_data = np.array(transformed_test_data)



# sess = tf.Session()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.46)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# Load the trained model from the directory "./model"

print(model_name+'/my_model.meta')
print(model_name)
saver = tf.train.import_meta_graph(model_name+'/my_model.meta')
saver.restore(sess, tf.train.latest_checkpoint(model_name))


# Load the model components
x = tf.get_collection('x')[0]
y_true = tf.get_collection('y_true')[0]
y_pred = tf.get_collection('y_pred')[0]
cost      = tf.get_collection('cost')[0]
optimizer = tf.get_collection('optimizer')[0]




# test_dicstart, test_diclength = handle_data.group(test_data)

# test_group_num = len(test_dicstart)

# general test ---------------------------------------------------------------

# current_test_data = handle_data.transform_data_to_test_form_data(test_data, negative_data)
# general_results = sess.run(y_pred, feed_dict={x: current_test_data})
general_results = sess.run(y_pred, feed_dict={x: test_data})
print(general_results)
general_pred_results = general_results[:,0].reshape(-1,1)

general_pred_results[general_pred_results<0.5] = 0
general_pred_results[general_pred_results>=0.5] = 1

# for i in range(test_time):
#     current_test_data = handle_data.transform_data_to_test_form_data(test_data, negative_data)
#     test_result = sess.run(y_pred, feed_dict={x: current_test_data})
#     test_pred_results = test_result[:, 0].reshape(-1,1)
#     test_pred_results[test_pred_results<0.5] = 0
#     test_pred_results[test_pred_results>=0.5] = 1
#     general_pred_results = np.hstack((general_pred_results, test_pred_results))


# calculate the sum of the 11 times test results, if sum > 5, it means that at least 6 times get the positive results, and the final result is positive, on the contrary, if sum < 6, it means that no more than 5 times get the positive result, the final result should be negative
# print(general_pred_results)
# general_vote_results = np.sum(general_pred_results, axis=1)
# print(general_vote_results)
# general_vote_results[general_vote_results<6] = 0
# general_vote_results[general_vote_results>5] = 1

# general_vote_results = general_vote_results.reshape(-1,1)

# print(general_vote_results)
print(general_pred_results)

file_data['predict_result'] = general_pred_results
all_file_data = file_data.values
file_data = pd.DataFrame(all_file_data)
file_data.to_csv(record_name, index=False)
print('Done')

true_label = test_label
predict_label = general_pred_results

all_group_top_precision = skmet.precision_score(y_true=true_label, y_pred=predict_label)
all_group_recall = skmet.recall_score(y_true=true_label, y_pred=predict_label)
all_group_fscore = skmet.f1_score(y_true=true_label, y_pred=predict_label)
all_group_auc = skmet.roc_auc_score(y_true=true_label, y_score=predict_label)
all_group_top_exact_accuracy = 1
all_group_exact_accuracy = 1
earn_rate = 1

record_name = record_name[0:-4] + '.txt'

predict_test.cal_average(all_group_top_precision, all_group_recall, all_group_fscore, all_group_top_exact_accuracy, all_group_exact_accuracy, all_group_auc, earn_rate, record_name)