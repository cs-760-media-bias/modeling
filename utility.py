import numpy as np
import os
from sklearn import metrics, preprocessing

IN_PATH = 'preprocessed'


def print_error(predicted, actual):
    mean_squared_error = metrics.mean_squared_error(actual, predicted)
    mean_absolute_error = metrics.mean_absolute_error(actual, predicted)
    median_absolute_error = metrics.median_absolute_error(actual, predicted)
    r2_score = metrics.r2_score(actual, predicted)
    mean_value = np.mean(actual)

    print('\tMean squared error:    ' + str(mean_squared_error))
    print('\tMean absolute error:   ' + str(mean_absolute_error))
    print('\tMedian absolute error: ' + str(median_absolute_error))
    print('\tR squared score:       ' + str(r2_score))
    print('\tMean label value:      ' + str(mean_value))


def print_results(X_train, X_test, y_train, y_test, model):
    print('Training set:')
    print_error(model.predict(X_train), y_train)
    print('Test set:')
    print_error(model.predict(X_test), y_test)
    print()


def load():
    print('Loading data...')
    X_train = np.loadtxt(os.path.join(IN_PATH, 'X_train.csv'), delimiter=',')
    X_test = np.loadtxt(os.path.join(IN_PATH, 'X_test.csv'), delimiter=',')
    y_train = np.loadtxt(os.path.join(IN_PATH, 'y_train.csv'), delimiter=',')
    y_test = np.loadtxt(os.path.join(IN_PATH, 'y_test.csv'), delimiter=',')

    # Use all the features except ad_fontes_y and ad_fontes_x
    X_train = X_train[:, 2:]
    X_test = X_test[:, 2:]

    # Use favorite_count as the label
    y_train = y_train[:, 3]
    y_test = y_test[:, 3]

    return X_train, X_test, y_train, y_test


def standardize(X_train, X_test):
    print('Standardizing data...')
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test
