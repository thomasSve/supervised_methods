import numpy as np
import operator
import math
from utils import *

"""
KNN: we use 26 values of K ranging from K = 1 to
K = |trainset|. We use KNN with Euclidean distance
and Euclidean distance weighted by gain ratio. We
also use distance weighted KNN, and locally weighted
averaging. The kernel widths for locally weighted averaging
vary from 2^0 to 2^10 times the minimum distance
between any two points in the train set.
"""
def euclidean_distance(instance1, instance2, length):
    distance = 0
    for i in range(length):
        distance += pow((instance1[i] - instance2[i]), 2)
    return math.sqrt(distance)

def get_neighbors(distances, k):
    distances.sort(key=operator.itemgetter(1))
    return [distances[i] for i in range(k)]

def predict_value(neighbors):
    # Find what the majority label is
    class_votes = {}
    for i in range(len(neighbors)):
        response = neighbors[i][0]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1

    sorted_values = sorted(class_votes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_values[0][0]

def test_kNN_classifier(X_train, y_train, X_test, y_test):
    k_values = [1, 5, 7, 12, 16, 26, 40, 60, 80, 100, 250, 500, 750, 1000, 2000, len(X_train)]
    #k_values = [1, 3, 5, 7]
    distances = calculate_distances(X_train, y_train, X_test, y_test)
    scores = []
    for k in k_values:
        predictions = []
        for i in range(len(X_test)):
            neighbors = get_neighbors(distances[i], k)
            predictions.append(predict_value(neighbors))
        scores.append((k, 1 - calculate_error(predictions, y_test)))
    return scores

def calculate_distances(X_train, y_train, X_test, y_test):
    # Calculate distances from all points to eachother
    len_train = len(X_train)
    len_test = len(X_test)
    
    distances = []
    for i in range(len_test):
        distances.append([])
        length = len(X_test[i])-1
        for j in range(len_train):
            dist = euclidean_distance(X_test[i], X_train[j], length)
            distances[i].append((y_train[j], dist))
    return distances

def run_kNN(X_train, y_train, X_test, y_test):
    scores = test_kNN_classifier(X_train, y_train, X_test, y_test)
    print("kNN (k-value, score)", scores)
