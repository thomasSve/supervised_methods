import numpy as np
from utils import save_generated_data
from collections import defaultdict

# Covertype
covtype = "datasets/covertype/covtype.data"
covtype_debug = "datasets/covertype/covtype_debug.data"

# Convert cov_type into a binary problem with largest class as positive and the rest as negative
def generate_covtype_data():
    # Create a root node for the tree
    X = []
    y = []
    with open(covtype, 'r') as file:
        i = 0
        for line in file:
            words = line.split(',')
            words = map(str.strip, words)
            X.append(words[:-1])
            y.append(words[-1])
        file.close()
    X = np.array(X, dtype='int64')
    y = np.array(y, dtype='int64')
    return X, y

def largest_class(y):
    d = defaultdict(int)
    for i in y:
        d[i] += 1
    result = max(d.iteritems(), key=lambda x: x[1])
    for i in range(len(y)):
        if y[i] == (result[0]):
            y[i] = 1
        else:
            y[i] = -1
            
    print(result[0])

    return y

def main():
    X, y = generate_covtype_data()
    y = largest_class(y) # largest class as the positive and the rest as negative
    save_generated_data(X, y, "covtype_generated")
    
if __name__ == "__main__":
    main()
