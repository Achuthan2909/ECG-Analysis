from datasets import ECGSequence
import argparse
import matplotlib.pyplot as plt

def load_data(path_to_hdf5, dataset_name, path_to_csv, val_split=0.2, batch_size=1):
    train_seq, valid_seq = ECGSequence.get_train_and_val(
        path_to_hdf5, dataset_name, path_to_csv, batch_size, val_split
    )
    return train_seq, valid_seq

def plot_first_batch(train_seq):
    # Print values of the first batch
    batch_idx = 0
    batch_x, batch_y = train_seq[batch_idx]
    reshaped_data = batch_x[0,:,]  # Taking the first feature from each timestep
    print(f"Shape of reshaped data: {reshaped_data.shape}")
    for i in range(reshaped_data.shape[1]):
        plt.plot(reshaped_data[:, i], label=f'Original Lead {i + 1}')
        plt.show()

    print(f"Batch {batch_idx + 1} - Input Values:\n{batch_x}")
    print(f"Batch {batch_idx + 1} - Label Values:\n{batch_y}")

if __name__ == "__main__":
    # Get data and train
    parser = argparse.ArgumentParser(description='Train neural network.')
    parser.add_argument('path_to_hdf5', type=str,
                        help='path to hdf5 file containing tracings')
    parser.add_argument('path_to_csv', type=str,
                        help='path to csv file containing annotations')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='number between 0 and 1 determining how much of'
                             ' the data is to be used for validation. Default: 0.2')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                        help='name of the hdf5 dataset containing tracings')
    args = parser.parse_args()
    batch_size = 1
    train_seq, _ = load_data(args.path_to_hdf5, args.dataset_name, args.path_to_csv, args.val_split, batch_size)

    # Plot the first batch
    plot_first_batch(train_seq)
