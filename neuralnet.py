from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import glob

import numpy as np
from numpy import genfromtxt


def norm(x, train_stats):
  return (x - train_stats['mean']) / train_stats['std']

def nn():

    directory = 'tweets_processed/abc.csv'

    column_names = ['video_count', 'photo_count', 'reply_to_user_id', 'text', 'created_at', 'hashtags', 'reply_to_tweet_id', 'user_mentions', 'urls', 'reply_to_screen_name', 'retweet_count', 'tweet_id', 'favorite_count', 'statuses_count', 'description', 'friends_count', 'account_created_at', 'followers_count', 'screen_name', 'listed_count', 'id', 'name']

    raw_dataset = pd.read_csv(directory, names=column_names)
    dataset = raw_dataset.copy()
    print(dataset.tail())

    # to do: one hot encoding 

    # split into train and test
    train_dataset = dataset.sample(frac=0.8,random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    # to do:pairwise plots

    # get train stats
    train_stats = train_dataset.describe()
    train_stats.pop("favorite_count")
    train_stats = train_stats.transpose()
    print(train_stats)

    # get tables
    train_labels = train_dataset.pop('favorite_count')
    test_labels = test_dataset.pop('favorite_count')

    # to do: normalize
    normed_train_data = norm(train_dataset, train_stats)
    normed_test_data = norm(test_dataset, train_stats)

    print(normed_train_data.tail())

    # build model
    model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])

    model.summary()
       
    example_batch = normed_train_data[:10]
    example_result = model.predict(example_batch)
    print(example_result)

    # train model
    EPOCHS = 5

    history = model.fit(
        normed_train_data, train_labels,
        epochs=EPOCHS, validation_split = 0.2, verbose=0)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())

    # test model
    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

    print("Testing set Mean Abs Error: {:5.2f}".format(mae))
    

if __name__== "__main__":
  nn()

