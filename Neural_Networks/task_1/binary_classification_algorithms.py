
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def preprocess():
    data = pd.read_csv('./dataset/dry_bean_dataset.csv')
    data['MinorAxisLength'].interpolate(method='linear', inplace=True)
    df = pd.DataFrame(data)
    return df


def single_preceptron(learning_rate, epochs, Mse_threshold, features, classes, bias=False):
    data = preprocess()
    data = data[(data['Class'] == 'BOMBAY') | (data['Class'] == 'SIRA')]
    X = data[features].values
    y = np.where(data['Class'] == classes[0], -1, 1)
    np.random.seed(0)
    weights = np.random.rand(X.shape[1])
    biass = np.random.rand() if bias else 0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    # Perceptron learning
    for epoch in range(epochs):
        errors = 0
        for xi, target in zip(X_train, y_train):
            prediction = np.sign(np.dot(xi, weights) + biass)
            # prediction= 0 if prediction <=0 else 1
            if target != prediction:
                weights += learning_rate * (target - prediction) * xi
                biass += learning_rate * (target - prediction)
                errors += 1
        if errors == 0:
            break

    # Calculate accuracy on the test set
    error = 0
    for i in range(len(y_test)):
        prediction = np.sign(np.dot(xi, weights)+biass)
        prediction = -1 if prediction <= 0 else 1
        if prediction == y_test[i]:
            continue
        else:
            error += 1
    accuracy = 1 - (error / len(y_test))
    print(accuracy)
    return weights, biass, X_train, X_test, y_train, y_test


# def train_model(learning_rate,epochs,Mse_threshold,features,classes,bias=False):
# pass


def plotting(weights, bias, X_train, X_test, y_train, y_test):
    plt.scatter(X_test[y_test == 0][:, -1],
                X_test[y_test == 0][:, 1], label=f'Class BOMBAY')
    plt.scatter(X_test[y_test == 1][:, -1],
                X_test[y_test == 1][:, 1], label=f'Class SIRA')
    w0, w1 = weights
#   bias = bias_value

    x_range = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), num=100)
    y_range = (-bias - w0 * x_range) / w1
    plt.plot(x_range, y_range, label='Decision Boundary', linestyle='--')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()


weights, bias_value, X_train, X_test, y_train, y_test = single_preceptron(
    0.1, 100, 0.01, ['Area', 'Perimeter'], ['BOMBAY', 'SIRA'], bias=False)

plotting(weights, bias_value, X_train, X_test, y_train, y_test)


# def single_preceptron(learning_rate, epochs, Mse_threshold, features, classes, bias=False):
#   data = preprocess()
#   data = data[(data['Class'] == 'BOMBAY') | (data['Class'] == 'SIRA')]
#   X = data[features].values
#   y = np.where(data['Class'] == classes[0], 0, 1)
#   np.random.seed(0)
#   weights = np.random.rand(X.shape[1])
#   biass = np.random.rand() if bias else 0
#   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
#
#   # Perceptron learning
#   for epoch in range(epochs):
#       errors = 0
#       for xi, target in zip(X_train, y_train):
#           prediction = np.sign(np.dot(xi,weights)+biass)
#           #prediction= 0 if prediction <=0 else 1
#           if target != prediction :
#               weights += learning_rate * (target-prediction) * xi
#               biass += learning_rate * (target-prediction)
#               errors += 1
#       if errors == 0:
#           break
#   # Calculate accuracy on the test set
#         error=0
#   #accuracy = np.sum(y_test == np.sign(np.dot(X_test, weights) + biass)) / len(y_test)
#     for i in range(len(y_test)):
#         if np.sign(np.dot(X_test,weights) + biass) ==y_test:
#             continue
#         else:
#             error += 1
#     accuracy =  1- (error/len(y_test))
#     print(accuracy)
#     return weights, biass, X_train,X_test,y_train,y_test
