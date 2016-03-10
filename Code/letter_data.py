import numpy as np
from utils import save_generated_data

# Letter Recognition
letter = "datasets/letter-recognition/letter-recognition.data"
letter_debug = "datasets/letter-recognition/letter-recognition-debug.data"

# Letter uses letters A-M as positives and the rest as negatives
A_TO_M = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J','K','L','M']

def generate_letter_data():
    X = []
    y = []
    with open(letter_debug, 'r') as file:
        for line in file:
            words = line.split(',')
            words = map(str.strip, words)

            # Reading the values in as a dictionary
            X.append(words[1:])
            
            # Reading in the last value as a label (-1 og 1)
            if words[0] in A_TO_M :
                y.append(1)
            else:
                y.append(-1)
                
        file.close()

    X = np.array(X, dtype='int64')
    y = np.array(y, dtype='int64')
    return X, y

def main():
    X, y = generate_letter_data()
    save_generated_data(X, y, "letter_generated")
    
if __name__ == "__main__":
    main()
