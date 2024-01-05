import h5py
import math
import pandas as pd
from keras.utils import Sequence
import numpy as np

class ECGSequence(Sequence):
    @classmethod
    def get_train_val(cls, path_to_hdf5, hdf5_dset, path_to_csv=None, batch_size=8, val_split=0.2):
        # Calculate the total number of samples from the CSV file
        n_samples = len(pd.read_csv(path_to_csv))
        
        # Calculate the number of samples for training and validation
        n_train = math.ceil(n_samples * (1 - val_split))
        
        # Create training and validation sequences
        train_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, end_idx=n_train)
        val_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, start_idx=n_train)
        
        return train_seq, val_seq

    def __init__(self, path_to_hdf5, hdf5_dset, path_to_csv=None, batch_size=8,
                 start_idx=0, end_idx=None):
        # If no CSV file is provided, set y to None
        if path_to_csv is None:
            self.y = None
        else:
            # Read CSV file and get target values
            self.y = pd.read_csv(path_to_csv).values
        
        # Open the HDF5 file and get tracings
        self.f = h5py.File(path_to_hdf5, "r")
        self.x = self.f[hdf5_dset]
        
        # Set batch size and indices for data slicing
        self.batch_size = batch_size
        if end_idx is None:
            end_idx = len(self.x)
        self.start_idx = start_idx
        self.end_idx = end_idx

    @property
    def n_classes(self):
        # Return the number of classes (number of columns in y)
        return self.y.shape[1]

    def __getitem__(self, idx):
        # Define the start and end indices for data slicing
        start = self.start_idx + idx * self.batch_size
        end = min(start + self.batch_size, self.end_idx)
        
        if self.y is None:
            # If no target values, return only the input data
            return np.array(self.x[start:end, :, :])
        else:
            # Use the values of the first column of batch_y as target values for batch_x
            batch_x = np.array(self.x[start:end, :, :])
            batch_y = np.array(self.y[start:end])
            corrected_data = batch_y[:, 0]
            return batch_x, corrected_data

    def __len__(self):
        # Calculate the total number of batches
        return math.ceil((self.end_idx - self.start_idx) / self.batch_size)

    def __del__(self):
        # Close the HDF5 file when the object is deleted
        self.f.close()
