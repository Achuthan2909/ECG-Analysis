
import numpy as np
# Load predictions
predictions = np.load('/home/achu/Desktop/GitHubECGWork/Implementation_of_paper/ECG-Analysis/dnn_predicts/model.npy')
# Display the shape of the array
print("Shape of predictions array:", predictions.shape)
# Display the content of the array
print("Predictions array:", predictions)
