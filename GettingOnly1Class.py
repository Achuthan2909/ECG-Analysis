import argparse
from datasets import ECGSequence
import numpy as np

def load_data(path_to_hdf5, dataset_name, path_to_csv, val_split=0.2, batch_size=1):
    train_seq, valid_seq = ECGSequence.get_train_and_val(
        path_to_hdf5, dataset_name, path_to_csv, batch_size, val_split
    )
    return train_seq, valid_seq

if __name__ == "__main__":
    # Get data and train
    parser = argparse.ArgumentParser(description='Train neural network.')
    parser.add_argument('path_to_hdf5', type=str,
                        help='path to hdf5 file containing tracings')
    parser.add_argument('path_to_csv', type=str,
                        help='path to csv file containing annotations')
    parser.add_argument('--val_split', type=float, default=0.02,
                        help='number between 0 and 1 determining how much of'
                             ' the data is to be used for validation. The remaining '
                             'is used for validation. Default: 0.02')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                        help='name of the hdf5 dataset containing tracings')
    parser.add_argument('--batchSize', type=int, default=32,
                        help='batch size for training')  # Change type to int
    args = parser.parse_args()

    train_seq, valid_seq = load_data(args.path_to_hdf5, args.dataset_name, args.path_to_csv, args.val_split, args.batchSize)
    
    # Access the shape of the target data in the first batch
    first_batch_x, first_batch_y = train_seq[0]
    print(f"Shape of input data in the first batch: {first_batch_x.shape}")
    print(f"Shape of target data in the first batch: {first_batch_y.shape}")
