import tkinter as tk
from tkinter import ttk
from binary_classification_algorithms import train_model

def create_interface(root):
  root.title("Feature Selector")
  root.geometry("400x550")
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

  # Learning Rate
  ttk.Label(root, text="Enter learning rate (eta)").pack()
  eta_entry = ttk.Entry(root)
  eta_entry.pack(pady=(0, 5))

  # Number of Epochs
  ttk.Label(root, text="Enter number of epochs (m)").pack()
  epochs_entry = ttk.Entry(root)
  epochs_entry.pack(pady=(0, 5))

  # MSE Threshold
  ttk.Label(root, text="Enter MSE threshold (mse_threshold)").pack()
  mse_threshold_entry = ttk.Entry(root)
  mse_threshold_entry.pack(pady=(0, 5))

  # Bias Checkbox
  bias_var = tk.IntVar()
  bias_checkbox = ttk.Checkbutton(root, text="Add bias", variable=bias_var)
  bias_checkbox.pack(pady=(5, 0))

  # Algorithm Selection
  ttk.Label(root, text="Choose the used algorithm").pack()
  algorithm_var = tk.StringVar()
  algorithm_perceptron = ttk.Radiobutton(root, text="Perceptron", variable=algorithm_var, value="perceptron")
  algorithm_perceptron.pack()
  algorithm_adaline = ttk.Radiobutton(root, text="Adaline", variable=algorithm_var, value="adaline")
  algorithm_adaline.pack()

  # Train Button
  selected_features = get_selected_items(feature_listbox)
  selected_classes = get_selected_items(class_listbox)
  train_button = ttk.Button(root, text="Train Model", command=lambda: train_model(
        float(eta_entry.get()),
        int(epochs_entry.get()),
        float(mse_threshold_entry.get()),
        selected_features,
        selected_classes,
        bool(bias_var.get()),
        algorithm_var.get()
    ))
  train_button.pack(pady=(5, 0))
  
  root.mainloop()

# HELPER FUNCTIONS
def get_selected_items(listBox):
  selected_indices = listBox.curselection()
  selected_items = [listBox.get(idx) for idx in selected_indices]
  return selected_items