import numpy as np
import os
from sklearn import linear_model
from sklearn.model_selection import train_test_split

IN_PATH = 'preprocessed'

def print_error(X_train, X_test, y_train, y_test, model):
    print('Model weights:')
    print(model.coef_)

    err_train = np.mean(np.abs(model.predict(X_train) - y_train))
    mean_train = np.mean(y_train)
    print('Training set error:')
    print('\tMean absolute: \t' + str(err_train))
    print('\tMean value: \t' + str(mean_train))

    err_test = np.mean(np.abs(model.predict(X_test) - y_test))
    mean_test = np.mean(y_test)
    print('Test set error:')
    print('\tMean absolute: \t' + str(err_test))
    print('\tMean value: \t' + str(mean_test))

    print()


def ordinary_least_squares(X_train, X_test, y_train, y_test):
    print('Running ordinary least squares...')
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    print_error(X_train, X_test, y_train, y_test, model)


def ridge_regression(X_train, X_test, y_train, y_test, alphas, cv):
    print('Running ridge regression with ' +
          str(cv) + '-fold cross-validation...')
    print('Possible alpha values:')
    print(alphas)

    model = linear_model.RidgeCV(alphas=alphas, cv=cv)
    model.fit(X_train, y_train)
    print('Chosen alpha value: ' + str(model.alpha_))
    print_error(X_train, X_test, y_train, y_test, model)


def lasso_regression(X_train, X_test, y_train, y_test, alphas, cv):
    print('Running lasso regression with ' +
          str(cv) + '-fold cross-validation...')
    print('Possible alpha values:')
    print(alphas)

    model = linear_model.LassoCV(alphas=alphas, cv=cv)
    model.fit(X_train, y_train)
    print('Chosen alpha value: ' + str(model.alpha_))
    print_error(X_train, X_test, y_train, y_test, model)


if __name__ == '__main__':
    print('Loading data...')
    X_train = np.loadtxt(os.path.join(IN_PATH, 'X_train.csv'), delimiter=',')
    X_test = np.loadtxt(os.path.join(IN_PATH, 'X_test.csv'), delimiter=',')
    y_train = np.loadtxt(os.path.join(IN_PATH, 'y_train.csv'), delimiter=',')
    y_test = np.loadtxt(os.path.join(IN_PATH, 'y_test.csv'), delimiter=',')

    # Use all the features except ad_fontes_y and ad_fontes_x
    X_train = X_train[:,2:]
    X_test = X_test[:,2:]

    # Use favorite_count as the label
    y_train = y_train[:,3]
    y_test = y_test[:,3]

    ordinary_least_squares(X_train, X_test, y_train, y_test)

    ridge_alphas = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    ridge_regression(X_train, X_test, y_train, y_test, ridge_alphas, 10)

    lasso_alphas = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    lasso_regression(X_train, X_test, y_train, y_test, lasso_alphas, 10)
