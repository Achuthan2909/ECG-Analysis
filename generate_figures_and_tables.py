import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Read the CSV file
df = pd.read_csv('/home/achu/Desktop/Work/Train/filtered_exam.csv')

# Select only the first column as the target variable
y_true_binary = df.iloc[:, 0].astype(int)

# Load the NPY file
y_pred_model2 = np.load('/home/achu/Desktop/Work/ECG-Analysis/dnn_output.npy')

# Convert the predictions to binary (assuming it contains probabilities)
threshold = 0.5
y_pred_model2_binary = (y_pred_model2 > threshold).astype(int)

# Compute confusion matrix for model2
conf_matrix_model2 = confusion_matrix(y_true_binary, y_pred_model2_binary)

# Compute accuracy, precision, recall, and F1 score
accuracy_model2 = accuracy_score(y_true_binary, y_pred_model2_binary)
precision_model2 = precision_score(y_true_binary, y_pred_model2_binary)
recall_model2 = recall_score(y_true_binary, y_pred_model2_binary)
f1_model2 = f1_score(y_true_binary, y_pred_model2_binary)

# Print the results
print("Confusion Matrix for Model2:")
print(conf_matrix_model2)
print("\nAccuracy for Model2:", accuracy_model2)
print("Precision for Model2:", precision_model2)
print("Recall for Model2:", recall_model2)
print("F1 Score for Model2:", f1_model2)
