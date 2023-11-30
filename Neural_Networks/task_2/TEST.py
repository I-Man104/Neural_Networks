# %%
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import sys

# %%
def preprocess(classes, features):
   # Read the dataset
   data = pd.read_csv('./dataset/dry_bean_dataset.csv')
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
   data.insert(0, 'bias', 1)
   features.append("bias")
   X = data[features].values
   Y = np.where(data['Class'] == classes[0], -1, np.where(data['Class']==classes[1],0,1))
   x_train, x_test, y_train, y_test = train_test_split(
      X, Y, test_size=0.4, random_state=42)
   return x_train, x_test, y_train, y_test

# %%
x_train,x_test,y_train,y_test = preprocess(["BOMBAY","CALI","SIRA"],["Area","Perimeter","MajorAxisLength","MinorAxisLength","roundnes"])

x_train.shape

# %%
def sigmoidFunc(x):
    return (1/(1+np.exp(-x)))
def derivative_sigmoid(x):
    return x* (1-x)

# %%
def initialize_weights(hiddenlayers,n_neurons):
   weights = [np.random.rand(n_neurons[i], n_neurons[i+1]) for i in range(hiddenlayers+1)]
   return weights

# %%
weights = initialize_weights(2,[6,3,4,3])
weights

# %%
# Calculate neuron activation for an input
def activate_input(weights, inputs):
    activation = 0
    for i in range(len(weights.T)):
        for j in range(len(weights)):
            activation += weights[j] * inputs[i][j]
    activation=sigmoidFunc(activation)
    return activation


# %%
def activate_hidden(weights, activatedInput):
    activation = 0
    for i in range(len(weights)):
        for j in range(len(weights)):
            activation += weights[j] * activatedInput[i]
    activation=sigmoidFunc(activation)
    return activation

# %%
def backprop_out(y_train,activations):
    for i in y_train:
        output_layer_errors = activations[-1] - i
        delta = output_layer_errors * derivative_sigmoid(output_layer_errors)
    return delta

# %%
def backprop_hidden(delta,act):
    print(act)
    for j in range(len(act)-1):
        hidden_layer_error = np.dot(delta[j],act)
        hidden_delta = hidden_layer_error * derivative_sigmoid(hidden_layer_error)
    return hidden_delta

# %%
activations =[]
new_input = activate_input(weights[0], x_train)
activations.append(new_input)
for i in range(1, len(weights)):
    new_input = activate_hidden(weights[i], new_input)
    activations.append(new_input)
activations

# %%
Deltas =[]
delta = backprop_out(y_train,activations)
Deltas.append(delta)
activation =activations[:len(activations)-1]
for i in activation[::-1]:
    delta =backprop_hidden(delta,i)
    Deltas.append(delta)
Deltas

# %%

def update_weights(weights, learning_rate, delta):
   for i in range(len(weights)):
       weights[i] -= learning_rate * delta[i]
   return weights


# %%
weights_updated_out=[]

weights_updated_out.append(update_weights(weights,0.01,Deltas))
weights

# %%
epochs = 1000
def train(X,Y,epochs,learning,weights):
    for i in range(epochs):
        for count in range(len(X)):
            activations = []
            print(weights) 
            print("-------------")
            new_input = activate_input(weights[0], X)
            activations.append(new_input)
            for i in range(1, len(weights)):
                new_input = activate_hidden(weights[i], new_input)
                activations.append(new_input)
            target = [0]*3
            target[Y[count]] = 1
            Deltas =[]
            delta = backprop_out(target,activations)
            Deltas.append(delta)
            activation =activations[:len(activations)-1]
            for i in activation[::-1]:
                delta =backprop_hidden(delta,i)
                Deltas.append(delta)
            weights=update_weights(weights,learning,Deltas)
train(x_train,y_train,1000,0.01,weights)

# %%
hidden_layers = [3,4]
def train_acc(X,Y,hidden_layers):
    cnt = 0
    for count in range(len(X)):
        new_input = activate_input(weights[0], X)

        for i in range(1, len(weights)):
            new_input = activate_hidden(weights[i], new_input)

            if i == len(hidden_layers):
                mx = -1 * sys.float_info.max
                idx = -1

                for j in range(len(new_input)):
                    if new_input[j] > mx:
                        mx = new_input[j]
                        idx = j
                # Assuming y_train is a NumPy array
                    value_to_find = Y[count]
                    index_array = np.where(y_train == value_to_find)[0]

                    if index_array.size > 0:
                        true_class_index = index_array[0]

                        if true_class_index == idx:
                            cnt += 1
                    else:
                        print(f"Error: {value_to_find} not found in y_train array")
            

    print("Training acc= ", cnt / len(Y))
train_acc(x_train,y_train,hidden_layers)

# %%
def testing(x_test,y_test,hidden_layers):
    cnt = 0
    conf_matrix = np.zeros([3, 3], dtype=int)
    for count in range(len(y_test)):
        # Forward step
            new_input = activate_input(weights[0], x_test)
            for i in range(1, len(weights)):
                new_input = activate_hidden(weights[i], new_input)
                
                if i == len(hidden_layers)-1:
                    mx = -1 * sys.float_info.max
                    prediction = -1
                for j in range(len(new_input)):
                    if new_input[j] > mx:
                        mx = new_input[j]
                        prediction = j
                
                value_to_find = y_train[count]
                index_array = np.where(y_train == value_to_find)[0]

                if index_array.size > 0:
                    true_class_index = index_array[0]

                    if true_class_index == prediction:
                        cnt += 1
                else:
                    print(f"Error: {value_to_find} not found in y_train array")
    acc = cnt / len(y_test)
    print("Testing acc= ", cnt / len(y_test))
    return acc

# %%
testing(x_test,y_test,hidden_layers)


