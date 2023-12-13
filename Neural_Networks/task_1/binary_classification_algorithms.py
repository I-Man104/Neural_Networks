import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import evaluation
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Helper functions


def predict_model(X_test, y_test, W, bias):
    correct_predictions = 0
    for x_i, t_i in zip(X_test, y_test):
        y_i = np.dot(W, x_i) + bias
        if y_i < 0:
            y_i = -1
        else:
            y_i = 1
        if y_i == t_i:
            correct_predictions += 1

    actual_val = []
    for x in X_test:
        x = np.dot(W, x) + bias
        if x < 0:
            x = -1
            actual_val.append(x)
        else:
            x = 1
            actual_val.append(x)
    accuracy = correct_predictions / len(y_test)
    print(accuracy)
    return accuracy, actual_val


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
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.4, random_state=42)
    return x_train, x_test, y_train, y_test


def Plot(X, y, weights, bias):
    plt.scatter(X[:, 0], X[:, 1], c=y)

    # Plot the decision boundary
    x1 = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    x2 = -(weights[0]*x1 + bias) / weights[1]
    plt.plot(x1, x2, color='red')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Perceptron Decision Boundary')
    plt.show()

# SINGLE PERCEPTRON:


def perceptron(b, X, y, learning_rate=0.01, epochs=1000, use_bias=False):
    samples, feat = X.shape
    W = np.random.rand(feat)
    for _ in range(epochs):
        for i, x in enumerate(X):
            y_pred = np.dot(x, W) + b
            if y_pred >= 0:
                y_pred = 1
            else:
                y_pred = -1
            upt = learning_rate * (y[i] - y_pred)
            W += upt * x
            if use_bias:
                b += upt
    return W, b


def single_perceptron(learning_rate, epochs, features, classes, bias):
    x_train, x_test, y_train, y_test = preprocess(classes, features)
    W, bias = perceptron(0, x_train, y_train, learning_rate, epochs, bias)
    accuracy, actual_val = predict_model(x_test, y_test, W, bias)
    Plot(x_test, y_test, W, bias)
    return W, accuracy, actual_val, y_test

# ADALINE


def adaline(b, X, y, learning_rate=0.01, epochs=1000, use_bias=False, mse_threshold=0.01):
    samples, feat = X.shape
    W = np.random.rand(feat)
    total_squared_error = 0
    for _ in range(epochs):
        for i, x in enumerate(X):
            y_pred = np.dot(x, W) + b
            upt = learning_rate * (y[i] - y_pred)
            W += upt * x
            if use_bias:
                b += upt

            total_squared_error += (y[i] - y_pred) ** 2
            mse = total_squared_error / len(X)
            if mse <= mse_threshold:
                return W, b
    return W, b


def adaline_algorithm(learning_rate, epochs, features, classes, bias=False, mse_threshold=0.01):
    x_train, x_test, y_train, y_test = preprocess(classes, features)
    W, b = adaline(0, x_train, y_train, learning_rate,
                   epochs, bias, mse_threshold)
    accuracy, actual_val = predict_model(x_test, y_test, W, b)
    Plot(x_test, y_test, W, b)
    return W, accuracy, actual_val, y_test


def train_model(algorithm, learning_rate, epochs, features, classes, bias=False, mse_threshold=0.01):
    if algorithm == "perceptron":
        x = single_perceptron(learning_rate, epochs, features, classes, bias)
        actual_val = x[2]
        predicted = x[3]
        evaluation.Evaluation.plot_confusion_matrix(
            actual_val, predicted.tolist())
    else:
        x = adaline_algorithm(learning_rate, epochs,
                              features, classes, bias, mse_threshold)
        actual_val = x[2]
        predicted = x[3]
        evaluation.Evaluation.plot_confusion_matrix(
            actual_val, predicted.tolist())
