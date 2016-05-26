import numpy as np
from utils import save_generated_data

# Letter Recognition
letter = "datasets/letter-recognition/letter-recognition.data"
letter_debug = "datasets/letter-recognition/letter-recognition-debug.data"

# Letter uses letters A-M as positives and the rest as negatives
A_TO_M = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J','K','L','M']

def generate_letter_data():
    X = []
    y_p1 = []
    y_p2 = []
    with open(letter, 'r') as file:
        for line in file:
            words = line.split(',')
            words = map(str.strip, words)
            # Reading the values in as a dictionary
            X.append(words[1:])
            
            # Treats O as positive and rest as negative, gives unbalanced set
            if words[0] == 'O':
                y_p1.append(1)
            else:
                y_p1.append(-1)
            
            # Treats A-M as positive and rest as negative, gives a balanced set
            if words[0] in A_TO_M :
                y_p2.append(1)
            else:
                y_p2.append(-1)
                
        file.close()
        
    X = np.array(X, dtype='int64')
    y_p1 = np.array(y_p1, dtype='int64')
    y_p2 = np.array(y_p2, dtype='int64')
    return X, y_p1, y_p2

def main():
    X, y_p1, y_p2 = generate_letter_data()
    save_generated_data(X, y_p1, "letterp1_generated")
    save_generated_data(X, y_p2, "letterp2_generated")

if __name__ == "__main__":
    main()
