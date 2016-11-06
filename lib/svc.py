from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer


def model_test(x_data, y_data, cores=1):
    """
    model_test grid-searches through a bunch of

    :param x_data: scipy.sparse.csr_matrix
        Sparse matrix of NUM_SAMPLES x NUM_FEATURES containing feature data
    :param y_data: scipy.sparse.csr_matrix
        Sparse matrix of NUM_SAMPLES x 1 containing class data
    :param cores: int
        Number of cores to pass to GridSearchCV

    :return best_estimator_: sklearn.svm.SVC
        The best model that GridSearchCV can come up with

    """
    param_grid = {"C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20]}

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

    if y_train.count_nonzero() == 0 or y_test.count_nonzero() == 0:
        die_count = 0
        while y_train.count_nonzero() == 0 or y_test.count_nonzero() == 0:
            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
            die_count += 1
            if die_count > 100:
                print("Train/Test split failed")
                exit(0)

    print("Train/Test split complete:\n\tTrain: {} x {}, {} x {} [{}]\n\tTest: {} x {}, {} x {} [{}]".format(
        x_train.shape[0],
        x_train.shape[1],
        y_train.shape[0],
        y_train.shape[1],
        y_train.count_nonzero(),
        x_test.shape[0],
        x_test.shape[1],
        y_test.shape[0],
        y_test.shape[1],
        y_test.count_nonzero()))

    cv_skf = KFold(n_splits=3)
    svc_estimator = SVC(class_weight="balanced", kernel="linear", probability=True)
    f_scorer = make_scorer(fbeta_score, beta=2)

    grid_search = GridSearchCV(estimator=svc_estimator, param_grid=param_grid, cv=cv_skf, scoring=f_scorer,
                               n_jobs=cores, verbose=1)

    grid_search.fit(x_train, y_train.toarray().ravel())
    y_test_predict = grid_search.predict(x_test)

    for param in grid_search.best_params_.keys():
        print("Parameter {} Optimal: {}".format(param, grid_search.best_params_[param]))
    print(classification_report(y_test.toarray().ravel(), y_test_predict))

    return grid_search.best_estimator_
