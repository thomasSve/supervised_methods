import scipy.io
import numpy as np
from sklearn.grid_search import GridSearchCV
from math import floor
from random import shuffle


def save_generated_data(X, y, filename):
    np.savez("generated_arrays/" + filename, X=X, y=y)

def load_generated_data(filename):
    data = np.load("generated_arrays/" + filename + ".npz")
    return data['X'], data['y']

def grid_search_cross_val(classifier, params, X, y):
    clf = GridSearchCV(estimator=classifier, param_grid=params, cv=5)
    clf.fit(X, y)
    return clf

def split_into_train_test(X, y, train_size):
    data = zip(X, y)
    shuffle(data)
    X, y = zip(*data)
    
    return X[:train_size], X[train_size:], y[:train_size], y[train_size:]

def calculate_error(predictions, actual_values):
    errors = 0
    for i in range(len(predictions)):
        if predictions[i] != actual_values[i]:
            errors += 1
    return errors / float(len(predictions))

