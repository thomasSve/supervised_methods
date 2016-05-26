import numpy as np
from utils import *
#from k_nearest_neighbors import run_kNN
from k_nearest_sklearn import run_kNN_sklearn
from decision_tree import run_decisiontree
from svm import run_svm
#from neural_networks import run_neural_nets
from neural_network_pybrain import run_neural_network
from random_forrest import run_random_forrest
from boosted_trees import run_boosted_trees
from boosted_stumps import run_boosted_stumps
        
def main():
    
    # In order to save computational-time, the data is preprocessed
    datasets = []
    datasets.append(load_generated_data("adult_generated"))
    datasets.append(load_generated_data("letterp1_generated"))
    datasets.append(load_generated_data("letterp2_generated"))
    datasets.append(load_generated_data("covtype_generated"))

    experiments = ['Adult', 'Letter.p1', 'Letter.p2', 'Covertype']

    for i in range(len(datasets)):
        print("-------", experiments[i], "-----------")
        # Split dataset into train and test. Setting train-set to fixed 5000, and the rest to test
        X_train, X_test, y_train, y_test = split_into_train_test(datasets[i][0], datasets[i][1], train_size = 5000)
        # Run k_Nearest_Neighbors
        #run_kNN(X_train, y_train, X_test, y_test)
        run_kNN_sklearn(X_train, y_train, X_test, y_test)

        # Run decision_tree http://scikit-learn.org/stable/modules/tree.html
        run_decisiontree(X_train, y_train, X_test, y_test)

        # Run support_vector_machine http://scikit-learn.org/stable/modules/svm.html
        #run_svm(X_train, y_train, X_test, y_test)

        # Run neural_nets http://pybrain.org/docs/tutorial/fnn.html
        #run_neural_network(X_train, y_train, X_test, y_test)

        # Run random_forrest http://scikit-learn.org/stable/modules/ensemble.html#random-forests
        run_random_forrest(X_train, y_train, X_test, y_test)

        # Run boosted trees
        run_boosted_trees(X_train, y_train, X_test, y_test)

        # Run boosted stumps using Adaboost
        run_boosted_stumps(X_train, y_train, X_test, y_test)

        print(" ")
        

    
if __name__ == "__main__":
    main()
