import numpy as np
from utils import *
from k_nearest_neighbors import run_kNN
from decision_tree import run_decisiontree
from svm import run_svm
#from neural_networks import run_neural_nets
from random_forrest import run_random_forrest

def main():
    #X_cov_type, y_cov_type = generate_covtype_data(covtype_debug)
    #X_letter, y_letter = generate_letter_data(letter_debug)

    # In order to save computational-time, the data is preprocessed
    X, y = load_generated_data("adult_generated")
    # Split dataset into train and test. Setting train-set to fixed 5000, and the rest to test
    X_train, X_test, y_train, y_test = split_into_train_test(X, y, train_size = 30)

    # Run k_Nearest_Neighbors
    run_kNN(X_train, y_train, X_test, y_test)
    
    # Run decision_tree http://scikit-learn.org/stable/modules/tree.html
    run_decisiontree(X_train, y_train, X_test, y_test)
    
    # Run support_vector_machine http://scikit-learn.org/stable/modules/svm.html
    #run_svm(X_train, y_train, X_test, y_test)
    
    # Run neural_nets http://scikit-learn.org/dev/modules/neural_networks_supervised.html
    #run_neural_nets(X_train, y_train, X_test, y_test)
    
    # Run random_forrest http://scikit-learn.org/stable/modules/ensemble.html#random-forests
    run_random_forrest(X_train, y_train, X_test, y_test)
    
    # Run boosting_family_classifiers
    #run_boosting(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
