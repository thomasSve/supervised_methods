"""
Random Forests (RF): we tried both the BreimanCutler
and Weka implementations; Breiman-Cutler
yielded better results so we report those here. The
forests have 1024 trees. The size of the feature set
considered at each split is 1,2,4,6,8,12,16 or 20
"""

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from utils import *

def build_random_forrest_clf(X, y):
    clf = RandomForestClassifier(n_estimators=10, max_depth=None,  min_samples_split=1, random_state=0)
    clf = clf.fit(X, y)
    scores = cross_val_score(clf, X, y)
    print("Random forrest, training-score: ", scores.mean())
    return clf

def test_random_forrest_clf(X, y, clf):
    return clf.score(X, y)

def run_random_forrest(X_train, y_train, X_test, y_test):
    clf = build_random_forrest_clf(X_train, y_train)
    score = test_random_forrest_clf(X_test, y_test, clf)
    print("Random forrest, score: ", score)
