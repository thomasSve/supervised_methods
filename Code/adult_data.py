import numpy as np
from utils import save_generated_data

# Adult
adult = "datasets/adult/adult.data.txt"
adult_debug = "datasets/adult/adult_debug.data.txt"

# All possible values

values = {1: ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"],
          3: ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"],
          5: ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"],
          6: ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"],
          7: ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
          8: ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
          9: ["Female", "Male"],
          13: ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]}


# Define the categorial features
c_features = [1, 3, 5, 6, 7, 8, 9, 13]

def convert_features(line):
    for i in values.keys():
        for j in range(len(values[i])):
            if line[i] == values[i][j]:
                line[i] = j
                break
    return line

# Adult contains nominal attributes, converts them to boolean
def generate_adult_data():
    X = []
    y = []
    with open(adult_debug, 'r') as file:
        for line in file:
            words = line.split(',')
            words = map(str.strip, words)
            # If there's a missing value in the data, ignore the line
            if '?' in words:
                continue
            # Reading the values in as a dictionary
            features = convert_features(words[:-1])
            X.append(features)
            
            # Reading in the last value as a label (-1 og 1)
            if words[-1] == '<=50K':
                y.append(-1)
            else:
                y.append(1)
                
        file.close()

    X = np.array(X, dtype='int64')
    y = np.array(y, dtype='int64')

    # Encode the categorical features into integers    
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(categorical_features = c_features)
    X = enc.fit_transform(X).toarray()
    return X, y

def main():
    X, y = generate_adult_data()
    save_generated_data(X, y, "adult_generated")
    
if __name__ == "__main__":
    main()
