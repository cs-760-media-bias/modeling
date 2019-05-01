from sklearn import linear_model
import utility


def ordinary_least_squares(X_train, X_test, y_train, y_test):
    print('Running ordinary least squares...')
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    print('Model weights:')
    print(model.coef_)
    utility.print_results(X_train, X_test, y_train, y_test, model)


def ridge_regression(X_train, X_test, y_train, y_test, alphas, cv):
    print('Running ridge regression with ' +
          str(cv) + '-fold cross-validation...')
    print('Possible alpha values:')
    print(alphas)

    model = linear_model.RidgeCV(alphas=alphas, cv=cv)
    model.fit(X_train, y_train)
    print('Chosen alpha value: ' + str(model.alpha_))
    print('Model weights:')
    print(model.coef_)
    utility.print_results(X_train, X_test, y_train, y_test, model)


def lasso_regression(X_train, X_test, y_train, y_test, alphas, cv):
    print('Running lasso regression with ' +
          str(cv) + '-fold cross-validation...')
    print('Possible alpha values:')
    print(alphas)

    model = linear_model.LassoCV(alphas=alphas, cv=cv)
    model.fit(X_train, y_train)
    print('Chosen alpha value: ' + str(model.alpha_))
    print('Model weights:')
    print(model.coef_)
    utility.print_results(X_train, X_test, y_train, y_test, model)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = utility.load()
    X_train, X_test = utility.standardize(X_train, X_test)
    print()

    ordinary_least_squares(X_train, X_test, y_train, y_test)

    ridge_alphas = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    ridge_regression(X_train, X_test, y_train, y_test, ridge_alphas, 10)

    lasso_alphas = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    lasso_regression(X_train, X_test, y_train, y_test, lasso_alphas, 10)
