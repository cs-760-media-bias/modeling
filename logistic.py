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

    tweets = json_data['tweets']
    # get tweet attributes and values
    tweet_attributes = np.array([tweet for tweet in tweets[0]])
    tweet_values = np.array([[tweet[attribute] for attribute in tweet_attributes] for tweet in tweets])

    for i in range(len(tweet_values[0])):
        value = tweet_values[1][i]
        print(i,tweet_attributes[i],value)

    # delete attributes
    delete_cols = np.delete(tweet_values, [2,3,4,5,6,7,8,9,11], 1)
    delete_attributes = np.delete(tweet_attributes, [2,3,4,5,6,7,8,9,11], 0)
    for i in range(len(delete_cols[0])):
        value = delete_cols[1][i]
        print(i,delete_attributes[i],value)
    # to do: add user values to tweet values and user attributes to tweet attributes

    X = delete_cols[:,[0,1,2]]
    Y = delete_cols[:,[3]]

    train_index = np.random.choice(len(X), int(len(X) * 0.8), replace=False)
    test_index = np.array(list(set(range(len(X))) - set(train_index)))

    train_X = X[train_index]
    train_y = Y[train_index]
    test_X = X[test_index]
    test_y = Y[test_index]

    A = tf.Variable(tf.random_normal(shape=[3, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    data = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    target = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    mod = tf.matmul(data, A) + b
    loss = tf.reduce_mean(tf.square(mod - target))

    learning_rate = 0.00000001
    batch_size = 30
    iter_num = 3000

    opt = tf.train.GradientDescentOptimizer(learning_rate) 
    goal = opt.minimize(loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(iter_num):
            feed_dict={data: train_X, target:train_y}
            sess.run(goal, feed_dict)
            #if i%100 == 0:
            #    print(i, "loss:", loss.eval(feed_dict))
            test_feed_dict = {data: test_X, target:test_y}
            sess.run(loss, test_feed_dict)
            if i%100 == 0:
                print(i, "loss:", loss.eval(test_feed_dict))



if __name__== "__main__":
    main()
