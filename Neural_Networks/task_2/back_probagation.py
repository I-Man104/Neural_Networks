import pandas as pd
import numpy as np
import evaluation
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
from sklearn.utils import shuffle

def preprocess(classes, features,bias):
    # Read the dataset
    #./dataset/dry_bean_dataset.csv
    data = pd.read_csv("./dataset/dry_bean_dataset.csv")
    # Filter rows based on the 'Class' column
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
    if(bias):
        data.insert(0, 'bias', 1)
        features.append("bias")
    X = data[features].values
    Y = np.where(data['Class'] == classes[0], -1, np.where(data['Class']==classes[1],0,1))
    x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.4, random_state=42)
    return x_train, x_test, y_train, y_test

# x_train.shape
def sigmoidFunc(x):
    return (1/(1+np.exp(-x)))
def derivative_sigmoid(x):
    return x* (1-x)
def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
def tanh_derivative(x):
    tanh_x = tanh(x)
    return 1 - tanh_x**2

def initialize_weights(hiddenlayers,n_neurons):
    weights = [np.random.rand(n_neurons[i], n_neurons[i+1]) for i in range(hiddenlayers+1)]
    return weights

# Calculate neuron activation for an input
def activate_input(weights, inputs,sig):
    activation = 0
    for i in range(len(weights.T)):
        for j in range(len(weights)):
            activation += weights[j] * inputs[i][j]
    if sig:
        activation=sigmoidFunc(activation)
    else:
        activation = tanh(activation)
    return activation

def activate_hidden(weights, activatedInput,sig):
    activation = 0
    for i in range(len(weights)):
        for j in range(len(weights)):
            activation += weights[j] * activatedInput[i]
    if sig:
        activation=sigmoidFunc(activation)
    else:
        activation = tanh(activation)
    return activation

def backprop_out(y_train,activations,sig):
    for i in y_train:
        output_layer_errors = activations[-1] - i
        if sig:
            delta = output_layer_errors * derivative_sigmoid(output_layer_errors)
        else:
            delta = output_layer_errors * tanh_derivative(output_layer_errors)         
    return delta

def backprop_hidden(delta,act,sig): 
    for j in range(len(delta)):
        hidden_layer_error = np.dot(delta[j],act)
        if sig:
            hidden_delta = hidden_layer_error * derivative_sigmoid(hidden_layer_error)
        else:
            hidden_delta = hidden_layer_error * tanh_derivative(hidden_layer_error)

    return hidden_delta

def update_weights(weights, learning_rate, delta):
    for i in range(len(weights)):
       weights[i] -= learning_rate *  delta[i]
    return weights

def train(X,Y,epochs,learning,weights,sig):
    for i in range(epochs):
        for count in range(len(X)):
            activations = []
            new_input = activate_input(weights[0], X,sig)
            activations.append(new_input)
            for i in range(1, len(weights)):
                new_input = activate_hidden(weights[i], new_input,sig)
                activations.append(new_input)
            target = [0]*3
            target[Y[count]] = 1
            Deltas =[]
            delta = backprop_out(target,activations,sig)
            Deltas.append(delta)
            activation =activations[:len(activations)-1]
            for i in activation[::-1]:
                delta =backprop_hidden(delta,i,sig)
                Deltas.append(delta)
            weights=update_weights(weights,learning,delta)
    return weights


def train_acc(X,Y,hidden_layers,weights,sig):
    cnt = 0
    for count in range(len(X)):
        new_input = activate_input(weights[0], X,sig)

        for i in range(1, len(weights)):
            new_input = activate_hidden(weights[i], new_input,sig)

            if i == hidden_layers:
                mx = -1 * sys.float_info.max
                idx = -1

                for j in range(len(new_input)):
                    if new_input[j] > mx:
                        mx = new_input[j]
                        idx = j
                # Assuming y_train is a NumPy array
                    value_to_find = Y[count]
                    index_array = np.where(Y == value_to_find)[0]
                    if index_array.size > 0:
                        true_class_index = index_array[0]
                        if true_class_index == idx:
                            cnt += 1
                    else:
                        print(f"Error: {value_to_find} not found in y_train array")
            

    print("Training acc= ", cnt / len(Y))

def testing(y_train,x_test,y_test,hidden_layers,weights,activation_function):
    cnt = 0
    conf_matrix = np.zeros([3, 3], dtype=int)
    mx = 0
    for count in range(len(y_test)):
        # Forward step
            new_input = activate_input(weights[0], x_test,activation_function)
            for i in range(1, len(weights)):
                new_input = activate_hidden(weights[i], new_input,activation_function)
                
                if i == hidden_layers-1:
                    mx = -1 * sys.float_info.max
                    prediction = -1
                for j in range(len(new_input)):
                    if new_input[j] > mx:
                        mx = new_input[j]
                        prediction = j
                    # predictions.append(prediction)
                    conf_matrix[y_test,prediction]+=1
                
                value_to_find = y_train[count]
                index_array = np.where(y_train == value_to_find)[0]

                if index_array.size > 0:
                    true_class_index = index_array[0]

                    if true_class_index == prediction:
                        cnt += 1
                else:
                    print(f"Error: {value_to_find} not found in y_train array")
    acc = cnt / len(y_test)
    # conf=create_conf_matrix(y_test,prediction,hidden_layers)
    
    print("Testing acc= ", cnt / len(y_test))
    conf=conf_matrix.T
    print("confusion matrix= \n",conf)
    return acc,conf
def back_probagation_algo(features, classes, hidden_layers_num, neurons_num, learning_rate, epochs, activation_function, bias=False):
    x_train,x_test,y_train,y_test = preprocess(classes,features,bias)
    weights = initialize_weights(hidden_layers_num,neurons_num)
    weights = train(x_train,y_train,epochs,learning_rate,weights,activation_function)
    train_acc(x_train,y_train,hidden_layers_num,weights,activation_function)
    a,c = testing(y_train,x_test,y_test,hidden_layers_num,weights,activation_function)
    pass

def train_model(features, classes, hidden_layers_num, neurons_num, learning_rate, epochs, activation_function, bias=False):
    back_probagation_algo(features, classes, hidden_layers_num, neurons_num, learning_rate, epochs, activation_function, bias)
