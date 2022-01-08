from __future__ import absolute_import, division, print_function, unicode_literals
from IPython.display import clear_output

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf  # now import the tensorflow module
import os

basedir = os.path.abspath(os.path.dirname(__file__))
# Load dataset.
dftrain = pd.read_csv('http://storage.googleapis.com/tf-datasets/titanic/train.csv')  # training data
dfeval = pd.read_csv('http://storage.googleapis.com/tf-datasets/titanic/eval.csv')  # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']
# Create columns
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)

def make_input_fn(data_df, label_df, num_epochs=100, shuffle=True, batch_size=32):
    def input_function():  # inner function, this will be returned
        ds = tf.data.Dataset.from_tensor_slices(
            (dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
        if shuffle:
            ds = ds.shuffle(1000)  # randomize order of data
        ds = ds.batch(batch_size).repeat(
            num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
        return ds  # return a batch of the dataset

    return input_function  # return a function object for use
# Create stimator
train_input_fn = make_input_fn(dftrain,y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

def trainLinearRegression():
    # Use a breakpoint in the code line below to debug your script.
    print(tf.version)  # make sure the version is 2.x
    print(tf.__version__)
    linear_est.train(train_input_fn)  # train

def calculateLinearRegression():
    result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data
    clear_output()  # clears console output
    print(f'accuracy of the result {result["accuracy"]}')
    pred_dicts = list(linear_est.predict(eval_input_fn))
    probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
    colors = ['b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'y', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b']
    probs.plot(kind='hist', bins=20, title='predicted probabilities to survive in the Titanic by age', colors=colors)
    # seaborn.barplot(probs.index, probs.values)
    plt.savefig(basedir + '/static/images/new_plot.png')
    plt.show()
