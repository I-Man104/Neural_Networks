import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import evaluation
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def preprocess(classes, features):
    # Read the dataset
    data = pd.read_csv('./dataset/dry_bean_dataset.csv')
    # Filter rows based on the 'Class' column
    data = data[data['Class'].isin(classes)]
    # Perform linear interpolation for missing values in the 'MinorAxisLength' column
    data[features] = data[features].copy().fillna(data[features].mean())
    # Manually perform Min-Max scaling
    for column in features:
        min_val = data[column].min()
        max_val = data[column].max()
        data[column] = (data[column] - min_val) / (max_val - min_val)

    # Shuffle the data
    # data = data.sample(frac=1, random_state=0).reset_index(drop=True)
    data = shuffle(data, random_state=0)

    X = data[features].values
    Y = np.where(data['Class'] == classes[0], -1, 1)
    print(X, Y)
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.4, random_state=42)
    return x_train, x_test, y_train, y_test

def Plot(X, y, weights):
    plt.scatter(X[:, 0], X[:, 1], c=y)

    # Plot the decision boundary
    x1 = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    x2 = -(weights[0]*x1) / weights[1]
    plt.plot(x1, x2, color='red')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Perceptron Decision Boundary')
    plt.show()

def back_probagation_algo(features, classes, hidden_layers_num, neurons_num, learning_rate, epochs, activation_function, bias=False):
    
    pass

def train_model(features, classes, hidden_layers_num, neurons_num, learning_rate, epochs, activation_function, bias=False):
    back_probagation_algo(features, classes, hidden_layers_num, neurons_num, learning_rate, epochs, activation_function, bias=False)
