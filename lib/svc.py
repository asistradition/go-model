from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import normalize

from numpy.ma import MaskedArray


def model_test(x_data, y_data, cores=1, f_beta=0.5, param_grid=None, test_size=0.25, liblinear=False):
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
    :param liblinear: bool
        Use the LinearSVC implementation of liblinear instead of the SVC implementation of libsvm

    Return:

    :return best_estimator_: sklearn.svm.SVC
        The best SVC that GridSearchCV can come up with on the entire training data set
    :return cv_results_: dict
        The results of each iteration of the grid search. Can be converted to a TSV with the print_cv_results function

    """

    # Set default parameter grid if none was passed in
    if param_grid is None and not liblinear:
        param_grid = {"C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20], "kernel": ["linear"]}
    elif param_grid is None and liblinear:
        param_grid = {"C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20]}

    # Normalize features by sample & split


    # Build the estimator, crossvalidation, and scoring metric objects
    cv_skf = StratifiedShuffleSplit(n_splits=10, test_size=0.33)

    if liblinear:
        svc_estimator = LinearSVC(class_weight="balanced", loss="hinge")
    else:
        svc_estimator = SVC(class_weight="balanced")

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

    return grid_search.best_estimator_, grid_search.cv_results_


def model_build(x_data, y_data, test_size=None, kernel="rbf", C=None, gamma=None, degree=None):
    """
    Builds the SVC model using selected hyperparameters

    Required Arguments:

    :param x_data: scipy.sparse.csr_matrix
        Sparse matrix of NUM_SAMPLES x NUM_FEATURES containing feature data
    :param y_data: scipy.sparse.csr_matrix
        Sparse matrix of NUM_SAMPLES x 1 containing class data

    Keyword Arguments:

    :param test_size: float
        The ratio of test data to the entire data set. 1 - test_size is the training data set size.
        If None, then no test split will be performed and no scoring will occur
    :param kernel: str
        If linear, LinearSVC is used. Otherwise passed to SVC
    :param C, gamma, degree:
        If set, directly passed to the SVC. Otherwise the defaults are used

    :return svc_estimator:
    """

    if test_size is not None:
        x_train, x_test, y_train, y_test = safe_split(normalize(x_data), y_data, test_size=test_size,
                                                      print_results=True)
    else:
        x_train = normalize(x_data)
        y_train = y_data

    if kernel == "linear":
        svc_estimator = LinearSVC(class_weight="balanced", loss="hinge")
    else:
        svc_estimator = SVC(kernel=kernel, class_weight="balanced", probability=True)

    # Set attributes if passed
    if C is not None:
        setattr(svc_estimator, "C", C)
    if gamma is not None:
        setattr(svc_estimator, "gamma", gamma)
    if degree is not None:
        setattr(svc_estimator, "degree", degree)

    svc_estimator.fit(x_train, y_train.toarray().ravel())

    if test_size is not None:
        y_test_predict = svc_estimator.predict(x_test)
        print("\n" + classification_report(y_test.toarray().ravel(), y_test_predict))

    return svc_estimator


def safe_split(x_data, y_data, test_size=0.33, print_results=False):
    """
    Splits the data into training and test sets, requiring class examples for both classes in each set. Raises
    ValueError if this split fails

    Required Arguments:

    :param x_data: scipy.sparse.csr_matrix
        Sparse matrix of NUM_SAMPLES x NUM_FEATURES containing feature data
    :param y_data: scipy.sparse.csr_matrix
        Sparse matrix of NUM_SAMPLES x 1 containing class data

    Keyword Arguments:

    :param test_size: float
        The ratio of test data to the entire data set. 1 - test_size is the training data set size.
    :param print_results: bool
        Prints the results of the split to stdout

    Returns:

    :return x_train, x_test: scipy.sparse.csr_matrix
        Sparse matrix containing feature data
    :return y_train, y_test: scipy.sparse.csr_matrix
        Sparse matrix containing class data
    """

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
                raise ValueError("Train/Test split failed")

    if print_results:
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

    return x_train, x_test, y_train, y_test


def print_cv_results(cv_results, file):
    """
    Prints the grid_search CV results data structure to a TSV file.

    :param cv_results:
        GridSearchCV cv_results_ object (which is a dict of lists of stuff or something)
    :param file: file_handle
        File to print output to

    """
    params = cv_results.pop("params")

    keys = cv_results.keys()
    print("\t".join(keys), file=file)

    for key in keys:
        row_len = len(cv_results[key])
        if isinstance(cv_results[key], MaskedArray):
            cv_results[key] = list(map(str, cv_results[key].tolist("-")))
        else:
            cv_results[key] = list(map(str, cv_results[key]))

    for i in range(0, row_len):
        row = []
        for key in keys:
            row.append(cv_results[key][i])
        print("\t".join(row), file=file)
