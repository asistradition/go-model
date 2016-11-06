from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer


def model_test(x_data, y_data, cores=1, f_beta=2, param_grid=None, test_size=0.25):
    """
    model_test grid-searches through a parameter space and scores the results by fbeta

    Required Arguments:

    :param x_data: scipy.sparse.csr_matrix
        Sparse matrix of NUM_SAMPLES x NUM_FEATURES containing feature data
    :param y_data: scipy.sparse.csr_matrix
        Sparse matrix of NUM_SAMPLES x 1 containing class data

    Keyword Arguments:

    :param cores: int
        Number of cores to pass to GridSearchCV
    :param f_beta: int
        The weight of precision in the harmonic mean of precision and recall as part of the fbeta scoring metric
    :param test_size: float
        The ratio of test data to the entire data set. 1 - test_size is the training data set size
    :param param_grid: dict
        Param_grid to give to the GridSearchCV method. See documentation for scikit-learn. Will set some basic defaults
        if not passed in

    Return:

    :return best_estimator_: sklearn.svm.SVC
        The best SVC that GridSearchCV can come up with on the entire training data set

    """

    # Set default parameter grid if none was passed in
    if param_grid is None:
        param_grid = {"C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20], "kernel": ["linear"]}

    # Split the data set into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size)

    # Check and make sure that the training and test sets contain both classes
    # Important to do for highly imbalanced data sets
    if y_train.count_nonzero() == 0 or y_test.count_nonzero() == 0:
        die_count = 0
        while y_train.count_nonzero() == 0 or y_test.count_nonzero() == 0:
            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size)
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

    # Build the estimator, crossvalidation, and scoring metric objects
    cv_skf = StratifiedKFold(n_splits=3)
    svc_estimator = SVC(class_weight="balanced", probability=True)
    f_scorer = make_scorer(fbeta_score, beta=f_beta)

    # Build the grid search object
    grid_search = GridSearchCV(estimator=svc_estimator, param_grid=param_grid, cv=cv_skf, scoring=f_scorer,
                               n_jobs=cores, verbose=2)

    # Run the grid_search on the training data
    # Pass the feature data in as a csr_matrix
    # Flatten the classification data first, don't pass it in as a column array
    grid_search.fit(x_train, y_train.toarray().ravel())

    # Use the refit all-training estimator to predict the outcome from the test feature set
    y_test_predict = grid_search.predict(x_test)

    # Print the model parameters selected and the test data results
    for param in grid_search.best_params_.keys():
        print("Parameter {} Optimal: {}".format(param, grid_search.best_params_[param]))

    print("\n" + classification_report(y_test.toarray().ravel(), y_test_predict))

    return grid_search.best_estimator_
