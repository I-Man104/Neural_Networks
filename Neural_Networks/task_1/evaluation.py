import tkinter as tk
from tkinter import ttk
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Evaluation:
    def __init__(self, actual,predicted):
        predicted = predicted.tolist()
        self.true_pos = 0
        self.true_neg = 0
        self.false_pos = 0
        self.false_neg = 0
        for i, value in enumerate(actual):
            if actual[i] == predicted[i]:
                if actual[i] == 1:
                    self.true_pos += 1
                else:
                    self.true_neg += 1
            else:
                if actual[i] == 1:
                    self.false_neg += 1
                else:
                    self.false_pos += 1

        return self.true_pos, self.true_neg, self.false_pos, self.false_neg

    def get_precision(self):
        return self.true_positive/(self.true_positive+self.false_positive)
    def get_recall(self):
        return self.true_positive/(self.true_positive+self.false_negative)
    def get_f_measure(self):
        precision = self.get_precision()
        recall = self.get_recall()
        return 2*precision*recall / (precision+recall)

    def plot_confusion_matrix(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots()
        cax = ax.matshow(cm)

        # Add colorbar
        plt.colorbar(cax)

        # Set labels for the x and y axis
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # Display the matrix values as text
        for i in range(len(cm)):
            for j in range(len(cm[0])):
                ax.text(j, i, str(cm[i, j]), va='center', ha='center')

            plt.title('Confusion Matrix')
            window = tk.Tk()
            window.title("Confusion Matrix")
            canvas = FigureCanvasTkAgg(fig, master=window)
            canvas.get_tk_widget().pack()
