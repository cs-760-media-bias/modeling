import numpy as np
import os
import utility

IN_PATH = 'preprocessed'


def predict_mean(y_train, y_test):
    print('Running mean-value prediction model...')
    mean = np.mean(y_train)
    print('Mean value: ' + str(mean))
    print('Training set:')
    utility.print_error(np.repeat(mean, len(y_train)), y_train)
    print('Test set:')
    utility.print_error(np.repeat(mean, len(y_test)), y_test)
    print()


if __name__ == '__main__':
    print('Loading data...')
    y_train = np.loadtxt(os.path.join(IN_PATH, 'y_train.csv'), delimiter=',')
    y_test = np.loadtxt(os.path.join(IN_PATH, 'y_test.csv'), delimiter=',')
    print()

    # Use favorite_count as the label
    y_train = y_train[:, 3]
    y_test = y_test[:, 3]

    predict_mean(y_train, y_test)