from sklearn import neighbors, model_selection
import utility


def knn_uniform(X_train, X_test, y_train, y_test, ks, cv):
    print('Running KNN with ' + str(cv) + '-fold cross-validation...')
    print('Possible k values:')
    print(ks)

    model = neighbors.KNeighborsRegressor(
        n_jobs=-1, algorithm='auto', weights='uniform')
    params = {'n_neighbors': ks}
    cv_search = model_selection.GridSearchCV(model, params, cv=cv, n_jobs=-1)
    cv_search.fit(X_train, y_train)
    print('Chosen k value: ' + str(cv_search.best_params_['n_neighbors']))
    utility.print_results(X_train, X_test, y_train, y_test, cv_search)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = utility.load()
    X_train, X_test = utility.standardize(X_train, X_test)
    print()

    uniform_ks = [1, 2, 3, 5, 10]
    knn_uniform(X_train, X_test, y_train, y_test, uniform_ks, 10)
