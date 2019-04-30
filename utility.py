import numpy as np
from sklearn import metrics

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
