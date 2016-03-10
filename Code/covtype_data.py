import numpy as np
from utils import save_generated_data

# Covertype
covtype = "datasets/covertype/covtype.data"
covtype_debug = "datasets/covertype/covtype_debug.data"

# Convert cov_type into a binary problem with largest class as positive and the rest as negative
def generate_covtype_data():
    # Create a root node for the tree
    X = []
    y = []
    with open(covtype_debug, 'r') as file:
        for line in file:
            words = line.split(',')
            words = map(str.strip, words)
            
            X.append(words[:-1])
            if words[-1] == '2':
                y.append(1)
            else:
                y.append(-1)

        file.close()
    X = np.array(X, dtype='int64')
    y = np.array(y, dtype='int64')
    return X, y

def main():
    X, y = generate_covtype_data()
    save_generated_data(X, y, "covtype_generated")
    
if __name__ == "__main__":
    main()
