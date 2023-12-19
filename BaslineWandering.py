from datasets import ECGSequence
import matplotlib.pyplot as plt
import numpy as np
import pybaselines

def load_data(path_to_hdf5, dataset_name, path_to_csv, val_split=0.2, batch_size=1):
    train_seq, valid_seq = ECGSequence.get_train_and_val(
        path_to_hdf5, dataset_name, path_to_csv, batch_size, val_split
    )
    return train_seq, valid_seq

def Arpls_method(train_seq):
    # Print values of the first batch
    for batch_idx in range(len(train_seq)):
        batch_x, batch_y = train_seq[batch_idx]
        for j in range(batch_x.shape[0]):
            reshaped_data = batch_x[j, :,]
            for i in range(reshaped_data.shape[1]):
                noisy_data = np.array(reshaped_data[:, i])
                baseline, param1 = pybaselines.whittaker.arpls(noisy_data)
                corrected_data = np.array(noisy_data) - np.array(baseline)
                reshaped_data[:, i] = corrected_data
            print(f"shape of reshaped data: {reshaped_data.shape}")
        
if __name__ == "__main__":
    # Set default file paths and parameters
    default_hdf5_path = "C:/Users/Admin/Desktop/data/ecg_tracings.hdf5"
    default_csv_path = "C:/Users/Admin/Desktop/data/annotations/cardiologist1.csv"
    default_val_split = 0.2

    # Load data
    batch_size = 64
    train_seq, _ = load_data(default_hdf5_path, 'tracings', default_csv_path, default_val_split, batch_size)

    # Apply ArPLS method
    Arpls_method(train_seq)
