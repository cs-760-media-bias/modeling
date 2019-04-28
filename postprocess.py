import numpy as np
from numpy import genfromtxt
import glob

def post():

    directory = 'tweets_processed/'

    compiled_train_X = [] 
    compiled_train_y = []
    compiled_test_X = []
    compiled_test_y = []


    for csv in glob.glob(directory + '*.csv'):
        data = genfromtxt(csv, delimiter=',')

        # col[12] favorite count as label
        y = data[:,[12]]
        X = np.delete(data,[12] , 1)
    
        labels = ['video_count', 'photo_count', 'reply_to_user_id', 'text', 'created_at', 'hashtags', 'reply_to_tweet_id', 'user_mentions', 'urls', 'reply_to_screen_name', 'retweet_count', 'id', 'favorite_count', 'statuses_count', 'description', 'friends_count', 'created_at', 'followers_count', 'screen_name', 'listed_count', 'id', 'name']

        labels = np.delete(labels, [12], 0)

        #for i in range(len(labels)):
        #    print(i, labels[i], X[0][i])

        # get train and test sets
        train_index = np.random.choice(len(X), int(len(X) * 0.8), replace=False)
        test_index = np.array(list(set(range(len(X))) - set(train_index)))

        train_X = X[train_index]
        train_y = y[train_index]
        test_X = X[test_index]
        test_y = y[test_index]

        # compile train and test sets for each source
        if len(compiled_train_X) == 0:
            compiled_train_X = np.copy(train_X)
            compiled_train_y = np.copy(train_y)
            compiled_test_X = np.copy(test_X)
            compiled_test_y = np.copy(test_y)
        else:
            compiled_train_X = np.concatenate((compiled_train_X, train_X), axis=0)
            compiled_train_y = np.concatenate((compiled_train_y, train_y), axis=0)
            compiled_test_X = np.concatenate((compiled_test_X, test_X), axis=0)
            compiled_test_y = np.concatenate((compiled_test_y, test_y), axis=0)

        print(len(compiled_train_X))
        print(len(compiled_train_X[0]))
    

if __name__== "__main__":
  post()
