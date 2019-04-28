import json
import numpy as np
#from pandas.io.json import json_normalize

import tensorflow as tf
from tensorflow.keras import layers

import datetime
import dateutil.parser

import glob
import os

def main(jsonfile):
    
    with open(jsonfile, "r") as read_file:
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
        # col[2] reply to user id 
        compiled_values[i][2] = 0 if compiled_values[i][2] == None else 1
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
   
    savepath = 'tweets_processed/' + os.path.splitext(os.path.basename(jsonfile))[0] + '.csv'
    np.savetxt(savepath, compiled_values, fmt='%5s', delimiter=",")


if __name__== "__main__":
    directory = 'tweets_tidy/'
    for jsonfile in glob.glob(directory + '*.json'):
        main(jsonfile)
