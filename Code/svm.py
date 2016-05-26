from sklearn import svm
from utils import *

"""
SVMs: we use the following kernels in SVMLight
(Joachims, 1999): linear, polynomial degree 2 & 3, radial
with width {0.001,0.005,0.01,0.05,0.1,0.5,1,2}. We
also vary the regularization parameter by factors of ten from 10^-7
to 10^3 with each kernel.
"""

def build_svm_classifier(X, y, kernel_type='linear'):
    clf = svm.SVC(kernel=kernel_type)
    param_grid = ""
    if kernel_type == 'linear':
        param_grid = {'C': np.logspace(-7, 3, base=10), 'kernel': ['linear']}
    elif kernel_type == 'rbf': param_grid = {'C': np.logspace(-7, 3, base=10), \
    'gamma': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2] , 'kernel': ['rbf']}

    clf = grid_search_cross_val(clf, param_grid, X, y)
    return clf

def test_svm_classifier(X, y, clf):
    return clf.score(X, y)

def run_svm(X_train, y_train, X_test, y_test):
    # Linear kernel
    clf_linear = build_svm_classifier(X_train, y_train)
    score_linear = test_svm_classifier(X_test, y_test, clf_linear)
    print("Score, linear: ", score_linear)

    # rbf kernel
    clf_rbf = build_svm_classifier(X_train, y_train, kernel_type='rbf')
    score_rbf = test_svm_classifier(x_test, y_test, clf_rbf)
    print("Score, rbf: ", score_rbf)
