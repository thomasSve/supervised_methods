from sklearn.neural_network import MLPClassifier

"""
ANN we train neural nets with gradient descent
backprop and vary the number of hidden units
{1,2,4,8,32,128} and the momentum {0,0.2,0.5,0.9}.
We halt training the nets at many different epochs
and use validation sets to select the best nets.
"""

def build_neural_network(X, y):
    # Class MLPClassifier implements a multi-layer perceptron (MLP) algorithm that trains using Backpropagation.

    clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf = clf.fit(X, y)
    return clf

def test_neural_network(X, y, clf):
    return clf.score(X, y)

def run_neural_nets(X_train, y_train, X_test, y_test):
    clf = build_neural_network(X_train, y_train)
    score = test_neural_network(X_test, y_test, clf)
    print("Score neural-network", score)
