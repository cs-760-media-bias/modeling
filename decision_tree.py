from sklearn import tree, model_selection
import utility


def regression_tree(X_train, X_test, y_train, y_test, max_leaf_nodes, cv):
    print('Running regression tree with ' + str(cv) + '-fold cross-validation')
    print('Possible max number of leaf nodes:')
    print(max_leaf_nodes)

    model = tree.DecisionTreeRegressor(random_state=1)
    params = {'max_leaf_nodes': max_leaf_nodes}
    cv_search = model_selection.GridSearchCV(model, params, cv=cv, n_jobs=-1)
    cv_search.fit(X_train, y_train)
    print('Chosen max number of leaf nodes: ' +
          str(cv_search.best_params_['max_leaf_nodes']))
    utility.print_results(X_train, X_test, y_train, y_test, cv_search)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = utility.load()
    X_train, X_test = utility.standardize(X_train, X_test)
    print()

    max_leaf_nodes = [2, 5, 10, 15, 20, 25, 30]
    regression_tree(X_train, X_test, y_train, y_test, max_leaf_nodes, 10)
