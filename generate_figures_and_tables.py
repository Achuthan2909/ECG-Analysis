import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def evaluate_predictions(true_labels, predicted_labels, threshold=0.5):
    # Convert the predictions to binary (assuming it contains probabilities)
    predicted_labels_binary = (predicted_labels > threshold).astype(int)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels_binary)

    # Compute accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(true_labels, predicted_labels_binary)
    precision = precision_score(true_labels, predicted_labels_binary)
    recall = recall_score(true_labels, predicted_labels_binary)
    f1 = f1_score(true_labels, predicted_labels_binary)

    return conf_matrix, accuracy, precision, recall, f1

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate model predictions.')
    parser.add_argument('csv_file', type=str, help='path to the CSV file containing true labels')
    parser.add_argument('npy_file', type=str, help='path to the NPY file containing predicted labels')
    parser.add_argument('--target_column', type=str, default='target', help='name of the target column in the CSV file')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for binary classification')

    args = parser.parse_args()

    # Read the CSV file
    df = pd.read_csv(args.csv_file)

    # Select the specified column as the target variable
    y_true_binary = df[args.target_column].astype(int)

    # Load the NPY file
    y_pred_model = np.load(args.npy_file)

    # Evaluate predictions
    conf_matrix_model, accuracy_model, precision_model, recall_model, f1_model = evaluate_predictions(
        y_true_binary, y_pred_model, threshold=args.threshold
    )

    # Print the results
    print("Confusion Matrix:")
    print(conf_matrix_model)
    print("\nAccuracy:", accuracy_model)
    print("Precision:", precision_model)
    print("Recall:", recall_model)
    print("F1 Score:", f1_model)
