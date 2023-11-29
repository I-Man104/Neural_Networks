import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# from task1.binary_classification_algorithms import preprocess

# data = pd.read_csv('./dry_bean_dataset.csv')
# Initialize input data, target output, learning rate, and weights

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

   X = data[features].values
   Y = np.where(data['Class'] == classes[0], -1, np.where(data['Class']==classes[1],0,1))
   x_train, x_test, y_train, y_test = train_test_split(
      X, Y, test_size=0.4, random_state=42)
   return x_train, x_test, y_train, y_test


def sigmoid(x):
   return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)
def tanh_activation(x):
   return np.tanh(x)
def tanh_derivative(x):
   tan=tanh_activation(x)
   return 1 - tan**2
def initialize_weights(hiddenlayers,n_neurons):
   weights = [np.random.rand(n_neurons[i-1], n_neurons[i]) for i in range(hiddenlayers+1)]
   return weights

def forward_step(inputs, weights,sig=False):
   outputs = [np.array(inputs)]  
   for i in range(len(weights)):
      inputs = np.dot(inputs, weights[i])
      if not sig:
         inputs = sigmoid(inputs)
      else:
         inputs = tanh_activation(inputs)
      outputs.append(np.array(inputs))
   return outputs

def backward_step(inputs, outputs, target, weights, learning_rate,sig=False):
   if not sig:
      # errors = [target - outputs[-1] * sigmoid_derivative(outputs[-1])]
      output_layer_error = outputs[-1] - target

      errors = output_layer_error *  sigmoid_derivative(outputs[-1])

   else:
      errors = [target - outputs[-1] * tanh_derivative(outputs[-1])]
   deltas = [errors[-1]]
   
   for i in range(len(outputs)-2, 0, -1):
      if not sig:
         output_layer_error = outputs[-1] - target
         output_layer_delta = output_layer_error * sigmoid_derivative(outputs[-1])
         hidden_layer_error = np.dot(output_layer_delta, weights[i].T)
         hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_error)
      else:
         errors.append(deltas[-1].dot(weights[i].T) * tanh_derivative(outputs[i].T))

      deltas.append(errors[-1])
   
   deltas.reverse()
   
   for i in range(len(weights)):
      weights[i] += learning_rate * outputs[i].T.dot(deltas[i])

   return weights

def train_neural_network(inputs, targets, layer_sizes, learning_rate, max_epochs):
   weights = initialize_weights(layer_sizes)
   for epoch in range(max_epochs):
      overall_error = 0

      for i in range(len(inputs)):
            input_data = np.array([inputs[i]])
            target_data = np.array([targets[i]])

            outputs = forward_step(input_data, weights)
            weights = backward_step(input_data, outputs, target_data, weights, learning_rate)

            overall_error += np.sum(0.5 * (target_data - outputs[-1]) ** 2)


   return overall_error, weights,outputs



X_train,X_test,Y_train,Y_test = preprocess(["BOMBAY","CALI","SIRA"],["Area","Perimeter","MajorAxisLength","MinorAxisLength","roundnes"])
n_neurons = [3,4]
hidden_layers =2
weights = initialize_weights(hidden_layers,n_neurons)
print(weights)
learning_rate = 0.01
Epochs  = 1000
def predict(inputs, trained_weights):
   outputs = forward_step(inputs, trained_weights)
   return outputs[-1]

def calculate_accuracy(model, X_test, y_test,trained_weights):
   predictions =predict(X_test,trained_weights)
   predictions = predictions.flatten()
   correct_predictions = sum(predictions == y_test)
   accuracy = correct_predictions / len(y_test)
   return accuracy
error, trained_weights, predicted_labels = train_neural_network(X_train, Y_train, layer_sizes, learning_rate, Epochs)
print(predicted_labels)
accuracy = calculate_accuracy(predicted_labels, X_train, Y_train,trained_weights)
print(accuracy)