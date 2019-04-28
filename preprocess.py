from datetime import datetime, timedelta
import json
import numpy as np
import os
from tweet2vec.tweet2vec import tokenize, Doc2Vec

IN_PATH = 'tweets_tidy'
OUT_PATH = 'preprocessed'

TIME_WINDOW_HOURS = 24
MOST_RECENT = None
TWEET2VEC_FILENAME = os.path.join('tweet2vec', 'models', 'doc2vec')
TWEET2VEC_MODEL = Doc2Vec(TWEET2VEC_FILENAME)

# Prep functions should return a list of one or more values
USER_FEATURES = [
    {
        'json_name': 'description',
        'prep': lambda description: [len(description)],
        'out_name': 'description_length'
    },
    {
        'json_name': 'created_at',
        'prep': lambda created_at: [days_since(created_at)],
        'out_name': 'days_since_creation'
    },
    {'json_name': 'followers_count'},
    {'json_name': 'friends_count'},
    {'json_name': 'listed_count'},
    {'json_name': 'statuses_count'}
]

# Prep functions should return a list of one or more values
TWEET_FEATURES = [
    {
        'json_name': 'created_at',
        'prep': lambda created_at: [hours_since(created_at)],
        'out_name': 'hours_since_post'
    },
    {
        'json_name': 'text',
        'prep': lambda text: [len(text)],
        'out_name': 'text_length'
    },
    {
        'json_name': 'hashtags',
        'prep': lambda hashtags: [len(hashtags)],
        'out_name': 'hashtag_count'
    },
    {
        'json_name': 'user_mentions',
        'prep': lambda user_mentions: [len(user_mentions)],
        'out_name': 'user_mention_count'
    },
    {
        'json_name': 'urls',
        'prep': lambda urls: [len(urls)],
        'out_name': 'url_count'
    },
    {'json_name': 'photo_count'},
    {'json_name': 'video_count'},
    {
        'json_name': 'reply_to_user_id',
        'prep': lambda reply_to_user_id: [0 if reply_to_user_id is None else 1],
        'out_name': 'is_reply'
    },
    {
        'json_name': 'text',
        'prep': lambda text: list(TWEET2VEC_MODEL.vectorize(text)),
        'out_name': 'tweet_embedding'
    }
]

# Prep functions should return a single value packaged as a list
LABEL_FEATURE = {
    'json_name': 'favorite_count'
}


def get_value(parent, feature):
    value = parent[feature['json_name']]
    if 'prep' in feature:
        return feature['prep'](value)
    else:
        return [value]


def parse_datetime(string):
    return datetime.strptime(string, '%a %b %d %X %z %Y')


def hours_since(string):
    datetime = parse_datetime(string)
    return (MOST_RECENT - datetime).total_seconds() / 3600


def days_since(string):
    datetime = parse_datetime(string)
    return (MOST_RECENT - datetime).days


def preprocess():
    with open('sources.json') as sources_file:
        sources = json.load(sources_file)['sources']

    print('Determining most recently read tweet...')
    for source in sources:
        for handle in source['twitter_handles']:
            in_filename = os.path.join(IN_PATH, handle + '.json')
            if not os.path.isfile(in_filename):
                continue
            with open(in_filename) as in_file:
                tweets_json = json.load(in_file)
            for tweet in tweets_json['tweets']:
                created_at = parse_datetime(tweet['created_at'])
                global MOST_RECENT
                if MOST_RECENT is None or created_at > MOST_RECENT:
                    MOST_RECENT = created_at

    print('Preprocessing tweets...')
    tweets = []
    for source in sources:
        for handle in source['twitter_handles']:
            in_filename = os.path.join(IN_PATH, handle + '.json')
            if not os.path.isfile(in_filename):
                continue
            with open(in_filename) as in_file:
                tweets_json = json.load(in_file)

            user = tweets_json['user']
            user_values = []
            for feature in USER_FEATURES:
                user_values += get_value(user, feature)

            for tweet in tweets_json['tweets']:
                try:
                    tweet_values = []
                    tweet_values += user_values
                    for feature in TWEET_FEATURES:
                        tweet_values += get_value(tweet, feature)
                    tweet_values += get_value(tweet, LABEL_FEATURE)
                    tweets.append(tweet_values)
                except:
                    continue

    data = np.array(tweets, dtype='float')
    return {
        'features': data[:, :-1],
        'labels': data[:, -1],
    }


if __name__ == '__main__':
    output = preprocess()
    print('Writing data...')
    for name, arr in output.items():
        out_filename = os.path.join(OUT_PATH, name + '.csv')
        np.savetxt(out_filename, arr)
