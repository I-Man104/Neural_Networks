import tkinter as tk
from tkinter import ttk
from back_probagation import train_model

def create_interface(root):
    root.title("Feature Selector")
    root.geometry("400x400")

    # Number of hidden layers
    ttk.Label(root, text="Enter Number of hidden layers").pack()
    hidden_layers_entry = ttk.Entry(root)
    hidden_layers_entry.insert(0, "2")
    hidden_layers_entry.pack(pady=(0, 5))

    # Number of neurons in each hidden layer
    ttk.Label(root, text="Enter Number of neurons in each hidden layer").pack()
    neurons_entry = ttk.Entry(root)
    neurons_entry.insert(0, "6 3 4 3")
    neurons_entry.pack(pady=(0, 5))

    # Learning Rate
    ttk.Label(root, text="Enter learning rate (eta)").pack()
    eta_entry = ttk.Entry(root)
    eta_entry.insert(0, "0.01")
    eta_entry.pack(pady=(0, 5))

    # Number of Epochs
    ttk.Label(root, text="Enter number of epochs (m)").pack()
    epochs_entry = ttk.Entry(root)
    epochs_entry.insert(0, "1000")
    epochs_entry.pack(pady=(0, 5))

    # Bias Checkbox
    bias_var = tk.IntVar()
    bias_checkbox = ttk.Checkbutton(root, text="Add bias", variable=bias_var)
    bias_checkbox.pack(pady=(5, 0))

    # activation function Selection
    ttk.Label(root, text="Choose the used activation function").pack()
    activation_function_var = tk.StringVar()
    activation_function_var.set("sigmoid")
    activation_function_segmoid = ttk.Radiobutton(
        root, text="Sigmoid", variable=activation_function_var, value="sigmoid")
    activation_function_segmoid.pack()
    activation_function_tanh = ttk.Radiobutton(
        root, text="Tanh", variable=activation_function_var, value="tanh")
    activation_function_tanh.pack()
    
    def train_button_click():
        selected_features = ["Area","Perimeter","MajorAxisLength","MinorAxisLength","roundnes"]
        selected_classes = ["BOMBAY", "CALI", "SIRA"]
        
        string_neurons = neurons_entry.get().split(' ')
        int_neurons = [int(element) for element in string_neurons]
        train_model(
            selected_features,
            selected_classes,
            int(hidden_layers_entry.get()),
            int_neurons,
            float(eta_entry.get()),
            int(epochs_entry.get()),
            activation_function_var.get(),
            bool(bias_var.get()),
        )

    # Train Button
    train_button = ttk.Button(root, text="Train Model", command=train_button_click)
    train_button.pack(pady=(5, 0))