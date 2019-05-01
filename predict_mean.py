import numpy as np
import utility


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
    X_train, X_test, y_train, y_test = utility.load()
    print()

    predict_mean(y_train, y_test)
