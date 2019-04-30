import numpy as np
import os
from sklearn import neighbors, preprocessing, model_selection
import utility

IN_PATH = 'preprocessed'


def print_results(X_train, X_test, y_train, y_test, model):
    print('Training set error:')
    utility.print_error(model.predict(X_train), y_train)
    print('Test set error:')
    utility.print_error(model.predict(X_test), y_test)
    print()


def knn_regression(X_train, X_test, y_train, y_test, ks, cv):
    print('Running KNN regression with ' +
          str(cv) + '-fold cross-validation...')
    print('Possible k values:')
    print(ks)

    # Change n_jobs according to your hardware (-1 is all cores)
    model = neighbors.KNeighborsRegressor(n_jobs=-1, algorithm='auto')
    params = {'n_neighbors': ks}
    cv_search = model_selection.GridSearchCV(model, params, cv=10, n_jobs=-1)
    cv_search.fit(X_train, y_train)
    print('Chosen k value: ' + str(cv_search.best_params_['n_neighbors']))
    print_results(X_train, X_test, y_train, y_test, cv_search)


if __name__ == '__main__':
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

    # Standardize data
    print('Standardizing data...')
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    print()

    knn_regression(X_train, X_test, y_train, y_test, [1, 2, 3, 5, 10], 10)
