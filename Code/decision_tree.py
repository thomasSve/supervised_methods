from sklearn import tree
from utils import *
"""
Decision trees (DT): we vary the splitting criterion,
pruning options, and smoothing (Laplacian or
Bayesian smoothing). We use all of the tree models
in Buntine's IND package (Buntine & Caruana, 1991):
BAYES, ID3, CART, CART0, C4, MML, and SMML.
We also generate trees of type C44LS (C4 with no
pruning and Laplacian smoothing), C44BS (C44 with
Bayesian smoothing), and MMLLS (MML with Laplacian
smoothing). See (Provost & Domingos, 2003) for
a description of C44LS.
"""

def build_decision_tree(X, y):
    # Decision tree using entropy
    clf = tree.DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(X, y)
    return clf

def test_decisiontree_classifier(clf, X, y):
    return clf.score(X, y)

def run_decisiontree(X_train, y_train, X_test, y_test):
    clf = build_decision_tree(X_train, y_train) 
    score = test_decisiontree_classifier(clf, X_test, y_test)
    print("Decisiontree, score: ", score)
