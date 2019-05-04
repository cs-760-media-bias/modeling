import numpy as np
from sklearn import neural_network
import utility


def mlp_regressor(X_train, X_test, y_train, y_test, n_layers, layer_size):
    print('Running neural network with ' + str(n_layers) +
          ' layers of ' + str(layer_size) + ' units...')
    layer_sizes = np.repeat(layer_size, n_layers)
    model = neural_network.MLPRegressor(
        hidden_layer_sizes=layer_sizes, random_state=1, early_stopping=True)
    model.fit(X_train, y_train)
    utility.print_results(X_train, X_test, y_train, y_test, model)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = utility.load()
    X_train, X_test = utility.standardize(X_train, X_test)
    print()

    mlp_regressor(X_train, X_test, y_train, y_test, 3, 100)
