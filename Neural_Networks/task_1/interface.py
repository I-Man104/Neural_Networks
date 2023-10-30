import tkinter as tk
from tkinter import ttk
from binary_classification_algorithms import train_model

def create_interface(root):
  root.title("Feature Selector")
  root.geometry("400x650")
  # Feature Selection
  ttk.Label(root, text="Select two features").pack(pady=(5, 0))
  
  feature_listbox = tk.Listbox(root, selectmode=tk.MULTIPLE, height=5)
  features = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "Roundness"]
  
  for feature in features:
    feature_listbox.insert(tk.END, feature)
  
  feature_listbox.pack(pady=(0, 5))
  # Class Selection
  ttk.Label(root, text="Select two classes").pack()
  
  class_listbox = tk.Listbox(root, selectmode=tk.MULTIPLE, height=3)
  classes = ["BOMBAY", "CALI", "SIRA"]
  
  for class_ in classes:
    class_listbox.insert(tk.END, class_)
  
  class_listbox.pack(pady=(0, 5))

  # Spacer
  ttk.Label(root, text="").pack()

  # Learning Rate
  ttk.Label(root, text="Enter learning rate (eta)").pack()
  eta_entry = ttk.Entry(root)
  eta_entry.pack(pady=(0, 5))

  # Spacer
  ttk.Label(root, text="").pack()

  # Number of Epochs
  ttk.Label(root, text="Enter number of epochs (m)").pack()
  epochs_entry = ttk.Entry(root)
  epochs_entry.pack(pady=(0, 5))

  # Spacer
  ttk.Label(root, text="").pack()

  # MSE Threshold
  ttk.Label(root, text="Enter MSE threshold (mse_threshold)").pack()
  mse_threshold_entry = ttk.Entry(root)
  mse_threshold_entry.pack(pady=(0, 5))

  # Spacer
  ttk.Label(root, text="").pack()

  # Bias Checkbox
  bias_var = tk.IntVar()
  bias_checkbox = ttk.Checkbutton(root, text="Add bias", variable=bias_var)
  bias_checkbox.pack(pady=(5, 0))

  # Spacer
  ttk.Label(root, text="").pack()

  # Algorithm Selection
  ttk.Label(root, text="Choose the used algorithm").pack()
  algorithm_var = tk.StringVar()
  algorithm_perceptron = ttk.Radiobutton(root, text="Perceptron", variable=algorithm_var, value="perceptron")
  algorithm_perceptron.pack()
  algorithm_adaline = ttk.Radiobutton(root, text="Adaline", variable=algorithm_var, value="adaline")
  algorithm_adaline.pack()

  # Spacer
  ttk.Label(root, text="").pack()

  # Train Button
  train_button = ttk.Button(root, text="Train Model", command=train_model)
  train_button.pack(pady=(5, 0))
  
  root.mainloop()
