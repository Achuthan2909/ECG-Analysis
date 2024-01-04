import argparse
from datasets import ECGSequence
import numpy as np
from noise_removal import NR
# Continue with the modified import statement
from Model1 import get_model

# Continue with the remaining import statements and the rest of your script
from keras.optimizers import Adam
from keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                             CSVLogger, EarlyStopping)

def load_data(path_to_hdf5, dataset_name, path_to_csv, val_split=0.2, batch_size=1):
    train_seq, valid_seq = ECGSequence.get_train_val(
        path_to_hdf5, dataset_name, path_to_csv, batch_size, val_split
    )
    # Apply ArPLS method to baseline correct the data
    train_seq, valid_seq = NR.baselinewander(train_seq), NR.baselinewander(valid_seq)
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
    args = parser.parse_args()

    # Optimization settings
    loss = 'binary_crossentropy'
    lr = 0.001
    batch_size = 64
    opt = Adam(lr)
    callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=7,
                                   min_lr=lr / 100),
                 EarlyStopping(patience=9,  # Patience should be larger than the one in ReduceLROnPlateau
                               min_delta=0.00001)]

    train_seq, valid_seq = load_data(args.path_to_hdf5, args.dataset_name, args.path_to_csv, args.val_split, batch_size)

    # If you are continuing an interrupted section, uncomment the line below:
    #   model = keras.models.load_model(PATH_TO_PREV_MODEL, compile=False)
    model = get_model(train_seq.n_classes)
    model.compile(loss=loss, optimizer=opt)

    # Create log
    callbacks += [TensorBoard(log_dir='./logs', write_graph=False),
                  CSVLogger('training.log', append=False)]  # Change append to true if continuing training

    # Save the BEST and LAST model
    callbacks += [ModelCheckpoint('/home/achu/Desktop/Work/ECG-Analysis/BackUpModels/backup_model_last1.hdf5'),
                  ModelCheckpoint('/home/achu/Desktop/Work/ECG-Analysis/BackUpModels/backup_model_best1.hdf5', save_best_only=True)]

    # Train neural network
    history = model.fit(train_seq,
                        epochs=50,
                        initial_epoch=0,  # If you are continuing an interrupted section change here
                        callbacks=callbacks,
                        validation_data=valid_seq,
                        verbose=1)

    # Save final result
    model.save("/home/achu/Desktop/Work/ECG-Analysis/FinalModels/final_model1.hdf5")
