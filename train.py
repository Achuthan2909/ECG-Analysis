# Import necessary libraries and modules
import argparse
import matplotlib.pyplot as plt
from datasets import ECGSequence
import numpy as np
from noise_removal import NR

# Import specific functions from model modules
from Model1 import get_model
from Model2 import get_custom_model
from Model3 import get_combined_model

# Import Keras modules for model training
from keras.optimizers import Adam
from keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                             CSVLogger, EarlyStopping)

# Define a function to load data from HDF5 file and CSV file
def load_data(path_to_hdf5, dataset_name, path_to_csv, val_split=0.2, batch_size=1):
    # Get training and validation sequences from ECGSequence class
    train_seq, valid_seq = ECGSequence.get_train_val(
        path_to_hdf5, dataset_name, path_to_csv, batch_size, val_split
    )
    
    # Apply ArPLS method to baseline correct the data using noise_removal module
    train_seq, valid_seq = NR.baselinewander(train_seq), NR.baselinewander(valid_seq)
    
    return train_seq, valid_seq

# Define a function to train a given model and save checkpoints
def train_model(model_func, model_filename_prefix):
    # Create the specified model using the provided function
    model = model_func()
    
    # Compile the model with specified loss function and optimizer
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    # Define callbacks for model training
    callbacks = [
        TensorBoard(log_dir='./logs', write_graph=False),
        CSVLogger('training.log', append=False)
    ]

    # Save the BEST and LAST model during training
    callbacks += [
        ModelCheckpoint(f'{model_filename_prefix}_last.hdf5'),
        ModelCheckpoint(f'{model_filename_prefix}_best.hdf5', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    ]

    # Train neural network
    history = model.fit(
        train_seq,
        epochs=50,
        initial_epoch=0,
        callbacks=callbacks,
        validation_data=valid_seq,
        verbose=1
    )

    # Plot training and validation accuracy graphs
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()

    # Save the plots with model name
    plt.savefig(f'./training_plots_{args.model}.png')
    plt.show()

    # Save final result
    model.save(f'{model_filename_prefix}_final.hdf5')

# Main execution when script is run
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train neural network.')
    parser.add_argument('path_to_hdf5', type=str, help='path to hdf5 file containing tracings')
    parser.add_argument('path_to_csv', type=str, help='path to csv file containing annotations')
    parser.add_argument('--val_split', type=float, default=0.02,
                        help='number between 0 and 1 determining how much of'
                             ' the data is to be used for validation. The remaining '
                             'is used for validation. Default: 0.02')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                        help='name of the hdf5 dataset containing tracings')
    parser.add_argument('--model', type=str, choices=['model1', 'model2', 'model3', 'all'], default='model1',
                        help='Choose which model to train. Options: model1, model2, model3, all. Default: model1')
    args = parser.parse_args()

    # Set optimization parameters
    loss = 'binary_crossentropy'
    lr = 0.001
    batch_size = 64
    opt = Adam(lr)

    # Load data for training and validation
    train_seq, valid_seq = load_data(args.path_to_hdf5, args.dataset_name, args.path_to_csv, args.val_split, batch_size)

    # Train the specified model or all models based on the command line argument
    if args.model == 'model1' or args.model == 'all':
        train_model(get_model, './backup_model1')
    
    if args.model == 'model2' or args.model == 'all':
        train_model(get_custom_model, './backup_model2')

    if args.model == 'model3' or args.model == 'all':
        train_model(get_combined_model, './backup_model3')
