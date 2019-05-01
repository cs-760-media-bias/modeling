from datetime import datetime
import json
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tweet2vec.tweet2vec import Doc2Vec

IN_PATH = 'tweets_tidy'
OUT_PATH = 'preprocessed'
MOST_RECENT = None
HISTORY_LENGTH = 100
RETWEET_HISTORY = []
FAVORITE_HISTORY = []
TWEET2VEC_FILENAME = os.path.join('tweet2vec', 'models', 'doc2vec')
TWEET2VEC_MODEL = Doc2Vec(TWEET2VEC_FILENAME)

# Prep functions should return a list of one or more values
SOURCE_FEATURES = [
    # Objectivity - higher is more fact-based
    {'json_name': 'ad_fontes_y'},
    # Bias - negative is left, positive is right
    {'json_name': 'ad_fontes_x'}
]

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
        'json_name': 'created_at',
        'prep': lambda created_at: [parse_datetime(created_at).hour],
        'out_name': 'hour_in_day'
    },
    {
        'json_name': 'created_at',
        'prep': lambda created_at: [parse_datetime(created_at).weekday()],
        'out_name': 'day_in_week',
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
    {
        'json_name': 'retweet_count',
        'prep': lambda retweet_count: [update_history(RETWEET_HISTORY, retweet_count)],
        'out_name': 'historical_retweet_count'
    },
    {
        'json_name': 'favorite_count',
        'prep': lambda favorite_count: [update_history(FAVORITE_HISTORY, favorite_count)],
        'out_name': 'historical_favorite_count'

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

# Prep functions should return a list of one or more values
SOURCE_LABELS = [
    # Objectivity - higher is more fact-based
    {'json_name': 'ad_fontes_y'},
    # Bias - negative is left, positive is right
    {'json_name': 'ad_fontes_x'}
]

# Prep functions should return a list of one or more values
USER_LABELS = []

# Prep functions should return a list of one or more values
TWEET_LABELS = [
    {'json_name': 'retweet_count'},
    {'json_name': 'favorite_count'}
]


def get_value(parent, feature):
    value = parent[feature['json_name']]
    if 'prep' in feature:
        return feature['prep'](value)
    else:
        return [value]


def get_name(feature):
    if 'out_name' in feature:
        return feature['out_name']
    else:
        return feature['json_name']


def parse_datetime(string):
    return datetime.strptime(string, '%a %b %d %X %z %Y')


def days_since(string):
    datetime = parse_datetime(string)
    return (MOST_RECENT - datetime).days


def hours_since(string):
    datetime = parse_datetime(string)
    return int((MOST_RECENT - datetime).total_seconds() / 3600)


def update_history(history, count):
    if len(history) < HISTORY_LENGTH:
        history.append(count)
        raise ValueError('insufficient history')
    else:
        output = np.mean(history)
        history.append(count)
        history.pop(0)
        return output


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
    X = []
    y = []
    for source in sources:
        for handle in source['twitter_handles']:
            in_filename = os.path.join(IN_PATH, handle + '.json')
            if not os.path.isfile(in_filename):
                continue
            with open(in_filename) as in_file:
                tweets_json = json.load(in_file)

            global RETWEET_HISTORY
            global FAVORITE_HISTORY
            RETWEET_HISTORY = []
            FAVORITE_HISTORY = []

            # Traverse the tweets in chronological order (assumes sorted input)
            tweets_json['tweets'].reverse()
            for tweet in tweets_json['tweets']:
                # Put this in a try block because prep functions may fail
                try:
                    X_values = []
                    for feature in SOURCE_FEATURES:
                        X_values += get_value(source, feature)
                    for feature in USER_FEATURES:
                        X_values += get_value(tweets_json['user'], feature)
                    for feature in TWEET_FEATURES:
                        X_values += get_value(tweet, feature)
                    X.append(X_values)

                    y_values = []
                    for label in SOURCE_LABELS:
                        y_values += get_value(source, label)
                    for label in USER_LABELS:
                        y_values += get_value(tweets_json['user'], label)
                    for label in TWEET_LABELS:
                        y_values += get_value(tweet, label)
                    y.append(y_values)
                except:
                    continue

    return np.array(X, dtype='float'), np.array(y, dtype='float')


if __name__ == '__main__':
    X, y = preprocess()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)

    X_header = []
    for feature in SOURCE_FEATURES:
        X_header.append(get_name(feature))
    for feature in USER_FEATURES:
        X_header.append(get_name(feature))
    for feature in TWEET_FEATURES:
        X_header.append(get_name(feature))

    y_header = []
    for label in SOURCE_LABELS:
        y_header.append(get_name(label))
    for label in USER_LABELS:
        y_header.append(get_name(label))
    for label in TWEET_LABELS:
        y_header.append(get_name(label))

    print('Writing data...')
    np.savetxt(os.path.join(OUT_PATH, 'X_train.csv'), X_train,
               delimiter=',', header=','.join(X_header))
    np.savetxt(os.path.join(OUT_PATH, 'X_test.csv'), X_test,
               delimiter=',', header=','.join(X_header))
    np.savetxt(os.path.join(OUT_PATH, 'y_train.csv'), y_train,
               delimiter=',', header=','.join(y_header))
    np.savetxt(os.path.join(OUT_PATH, 'y_test.csv'), y_test,
               delimiter=',', header=','.join(y_header))
