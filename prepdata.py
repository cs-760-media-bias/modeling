import json
import numpy as np
#from pandas.io.json import json_normalize

import tensorflow as tf
from tensorflow.keras import layers

import datetime
import dateutil.parser

def main():
    directory = 'tweets_tidy/abc.json'
    
    with open(directory, "r") as read_file:
        json_data = json.load(read_file)

    user = json_data['user']
    # get user attributes and values
    user_attributes = np.array([attribute for attribute in user])
    user_values = np.array([user[attribute] for attribute in user])
    #print(user_attributes)
    print(user_values)
    print(len(user_values))

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

    print("")
    print(compiled_attributes)
    for i in range(len(compiled_attributes)):
        print(i,compiled_attributes[i], compiled_values[0][i])
    
    # column updates
    download_time = dateutil.parser.parse(compiled_values[0][4])
   
    for i in range(num_tweets):
        # col[3] text to len(text)
        compiled_values[i][3] = len(compiled_values[i][3])
        # col[4] created at to hrs since download
        created_time = dateutil.parser.parse(compiled_values[i][4])
        diff = download_time - created_time
        compiled_values[i][4] = float(diff.seconds)/360

        # col[5] hashtags to len(hashtags)
        compiled_values[i][5] = len(compiled_values[i][5])

        # col[6] replay to tweet id 1 if replay 0 if none
        compiled_values[i][6] = 0 if compiled_values[i][6] == None else 1

        # col[7] user_mentions to len(user_mentions)
        compiled_values[i][7] = len(compiled_values[i][7])

        # col[8] urls to 1 if url 0 if none
        compiled_values[i][8] = len(compiled_values[i][8])

        # col[9] replay to screen name 1 if reply 0 if none 
        compiled_values[i][9] = 0 if compiled_values[i][9] == None else 1

        # col[14] description to len(description)
        compiled_values[i][14] = len(compiled_values[i][14])

        # col[16] created at to days since created
        account_created = dateutil.parser.parse(compiled_values[i][16])
        account_diff = download_time - account_created
        compiled_values[i][16] = float(account_diff.days)

        # col[18] screen name
        compiled_values[i][18] = len(compiled_values[i][18])

        # col[21] user name
        compiled_values[i][21] = len(compiled_values[i][21])
    
    for j in range(len(compiled_attributes)):
        print(compiled_attributes[j], compiled_values[0][j])
    np.savetxt("tweets_processed/abc.csv", compiled_values, fmt='%5s', delimiter=",")


def postprocess():    
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
