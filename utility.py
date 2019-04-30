import numpy as np

def print_error(predicted, actual):
    mean_absolute = np.mean(np.abs(predicted - actual))
    mean_value = np.mean(actual)
    print('\tMean absolute: \t' + str(mean_absolute))
    print('\tMean value: \t' + str(mean_value))
