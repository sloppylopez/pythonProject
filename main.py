# This is a sample Python script.
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf
# %tensorflow_version 2.x
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf  # now import the tensorflow module

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    print(tf.version)  # make sure the version is 2.x
    print(tf.__version__)
    # Load dataset.
    dftrain = pd.read_csv('http://storage.googleapis.com/tf-datasets/titanic/train.csv')  # training data
    dfeval = pd.read_csv('http://storage.googleapis.com/tf-datasets/titanic/eval.csv')  # testing data
    y_train = dftrain.pop('survived')
    y_eval = dfeval.pop('survived')
    CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                           'embark_town', 'alone']
    NUMERIC_COLUMNS = ['age', 'fare']

    feature_columns = []
    for feature_name in CATEGORICAL_COLUMNS:
        vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
        feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

    for feature_name in NUMERIC_COLUMNS:
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

    print(feature_columns)

    def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
        def input_function():  # inner function, this will be returned
            ds = tf.data.Dataset.from_tensor_slices(
                (dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
            if shuffle:
                ds = ds.shuffle(1000)  # randomize order of data
            ds = ds.batch(batch_size).repeat(
                num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
            return ds  # return a batch of the dataset

        return input_function  # return a function object for use

    train_input_fn = make_input_fn(dftrain,
                                   y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
    eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)
    linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
    linear_est.train(train_input_fn)  # train
    result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data

    clear_output()  # clears console output
    print(f'result {result["accuracy"]}')
    # print(dftrain.shape)
    # print(dftrain.describe)
    # print(dftrain.head)
    # print(y_train.head)
    # print(y_eval.head)
    # print(dftrain.age.hist(bins=20))
    # print(dftrain.sex.value_counts().plot(kind='barh'))
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
