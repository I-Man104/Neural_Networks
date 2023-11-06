
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def preprocess():
  data = pd.read_csv('D:/Neural_Networks/dry_bean_dataset.csv')
  data['MinorAxisLength'].interpolate(method='linear', inplace=True)
  df = pd.DataFrame(data)
  return df

def adaline_algorithm(feature1,feature2, class1, class2, learning_rate, max_epochs, bias=False, mse_threshold=0.01):
    # Prepare the data
    data=preprocess()
    
    X = data[features].values
    y = np.where(data['Class'] == class1, 1, -1)

    # Initialize weights and bias
    np.random.seed(0)
    weights = np.random.rand(X.shape[1])
    bias_value = np.random.rand() if bias else 0

    for epoch in range(max_epochs):
        errors = 0
        total_squared_error = 0

        for xi, target in zip(X, y):
            # Calculate the predicted output
            output = np.dot(xi, weights) + bias_value

            # Calculate the error (e = ti - yi)
            error = target - output

            # Update weights and bias
            weights += learning_rate * error * xi
            bias_value += learning_rate * error if bias else 0

            # Update the total squared error
            total_squared_error += error ** 2

        # Calculate the mean squared error
        mse = total_squared_error / len(X)

        if mse <= mse_threshold:
            print(f"Converged after {epoch + 1} epochs.")
            break

    return weights, bias_value

# Example usage:
features = ['Area', 'Perimeter']
class1 = 'BOMBAY'
class2 = 'CALI'
learning_rate = 0.1
max_epochs = 100

trained_weights, trained_bias = adaline_algorithm(features, class1, class2, learning_rate, max_epochs, bias=True)
print(trained_weights,trained_bias)