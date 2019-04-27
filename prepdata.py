import json
import numpy as np
#from pandas.io.json import json_normalize

import tensorflow as tf
from tensorflow.keras import layers

def main():
    directory = 'tweets_tidy/abc.json'
    
    with open(directory, "r") as read_file:
        json_data = json.load(read_file)

    user = json_data['user']
    # get user attributes and values
    user_attributes = np.array([attribute for attribute in user])
    user_values = np.array([user[attribute] for attribute in user])
    print(user_attributes)
    print(user_values)

    tweets = json_data['tweets']
    num_tweets = len(tweets)
    # get tweet attributes and values
    tweet_attributes = np.array([tweet for tweet in tweets[0]])
    tweet_values = np.array([[tweet[attribute] for attribute in tweet_attributes] for tweet in tweets])

    # peek tweet at attributes and values
    #for i in range(len(tweet_values[0])):
    #    value = tweet_values[1][i]
    #    print(i,tweet_attributes[i],value)

    # combine user and tweet values
    compiled_values = [[] for i in range(num_tweets)]
    for i in range(num_tweets):
        values = np.append(tweet_values[i],user_values)
        compiled_values[i] = values
    compiled_attributes = np.append(tweet_attributes, user_attributes)

    print(compiled_attributes)
    for i in range(len(compiled_attributes)):
        print(i,compiled_attributes[i],compiled_values[0][i])

    
    #print(compiled_values[0])
    #print(len(compiled_values[0]))

    # delete attributes
    delete_cols = np.delete(tweet_values, [2,3,4,5,6,7,8,9,11], 1)
    delete_attributes = np.delete(tweet_attributes, [2,3,4,5,6,7,8,9,11], 0)
    for i in range(len(delete_cols[0])):
        value = delete_cols[1][i]
        print(i,delete_attributes[i],value)
    # to do: add user values to tweet values and user attributes to tweet attributes

    X = delete_cols[:,[0,1,2]]
    labels = delete_cols[:,[3]]
    labels_Y = []
    for label in labels:
        labels_Y.append(label[0])
    labels_Y = np.array(labels_Y)
    labels_Y.reshape((1, -1))

    # random without replacement 80% train 20% test
    #train_index = np.random.choice(len(X), round(len(X) * 0.8), replace=False)
    train_index = np.random.choice(len(X), int(len(X) * 0.8), replace=False)
    test_index = np.array(list(set(range(len(X))) - set(train_index)))

    train_X = X[train_index]
    train_y = labels_Y[train_index]
    test_X = X[test_index]
    test_y = labels_Y[test_index]
    #test_y = tf.convert_to_tensor(test_y)

def ml():
    x = tf.placeholder(tf.float32, name='x')
    y = tf.placeholder(tf.float32, name='y')

    w = tf.Variable(tf.zeros([len(train_X),3]), name='W')
    b = tf.Variable(tf.zeros([1]), name='b')

    y_pred = tf.add(tf.multiply(w, x), b)

    loss = tf.reduce_mean(tf.square(y_pred - y))

    optimizer = tf.train.GradientDescentOptimizer(0.00001)
    train_op = optimizer.minimize(loss)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        feed_dict = {x: train_X, y: train_y}
		
        for i in range(10000):
            session.run(train_op, feed_dict)
            if i%1000 == 0:
                print(i, "loss:", loss.eval(feed_dict))
        final_w = session.run(w)
        final_b = session.run(b)
        #print(final_w)
        #print(final_b)
        #pred_y = session.run(test_y, feed_dict={x: test_X})
        #mse = tf.reduce_mean(tf.square(pred_y - test_y))
        #print(session.run(mse))

if __name__== "__main__":
    main()
