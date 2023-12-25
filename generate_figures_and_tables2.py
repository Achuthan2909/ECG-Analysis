import pandas as pd
import numpy as np
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score)
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from scipy.stats.distributions import chi2
from itertools import combinations

# Auxiliar functions
def get_scores(y_true, y_pred, score_fun):
    scores = []
    for name, fun in score_fun.items():
        scores.append(fun(y_true, y_pred))
    return np.array(scores)

def specificity_score(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred, labels=[0, 1])
    spc = m[0, 0] * 1.0 / (m[0, 0] + m[0, 1])
    return spc

def get_optimal_precision_recall(y_true, y_score):
    precision, recall, threshold = precision_recall_curve(y_true, y_score)
    f1_score = 2 * precision * recall / (precision + recall)
    index = np.argmax(f1_score)
    opt_precision = precision[index]
    opt_recall = recall[index]
    opt_threshold = threshold[index]
    return opt_precision, opt_recall, opt_threshold

# Constants
score_fun = {'Precision': precision_score,
             'Recall': recall_score, 'Specificity': specificity_score,
             'F1 score': f1_score}

diagnosis = '1dAVb'

# Read datasets
y_true = pd.read_csv('C:/Users/Admin/Desktop/GithubWork/ECG-Analysis/data/annotations/gold_standard.csv')[diagnosis].values
y_score_model2 = np.load("C:/Users/Admin/Desktop/GithubWork/ECG-Analysis/dnn_predicts/other_seeds/model2.npy")

# Plot precision recall curve
precision, recall, threshold = precision_recall_curve(y_true, y_score_model2)
f1_score = 2 * precision * recall / (precision + recall)

# Plot the Precision-Recall curve
plt.figure(figsize=(8, 8))
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (F1 = {np.max(f1_score):.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve for {diagnosis} (model2)')
plt.legend()
plt.show()

# Confusion matrix
y_pred_model2 = (y_score_model2 > threshold).astype(int)
conf_matrix_model2 = confusion_matrix(y_true, y_pred_model2)
conf_matrix_df = pd.DataFrame(conf_matrix_model2, index=['True Negative', 'True Positive'], columns=['Predicted Negative', 'Predicted Positive'])
print(f'Confusion Matrix for {diagnosis} (model2):\n{conf_matrix_df}')

# Other analysis or visualizations related to model2 can be added here based on your requirements.
