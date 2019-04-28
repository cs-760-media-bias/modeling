import numpy as np
import os
from sklearn import linear_model
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    print('Loading data...')
    X = np.loadtxt(os.path.join('preprocessed', 'features.csv'))
    y = np.loadtxt(os.path.join('preprocessed', 'labels.csv'))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)

    print('Running ordinary least squares model...')
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    mean_abs_error = np.mean(np.abs(predictions - y_test))
    mean_test_label = np.mean(y_test)
    print('Mean absolute error: \t' + str(mean_abs_error))
    print('Mean test label value: \t' + str(mean_test_label))
