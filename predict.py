import numpy as np
import warnings
import argparse

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

# Import Keras modules
from keras.models import load_model
from keras.optimizers import Adam

# Import ECGSequence class from datasets module
from datasets import ECGSequence

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Get performance on the test set from hdf5')
    parser.add_argument('path_to_hdf5', type=str, help='path to hdf5 file containing tracings')
    parser.add_argument('path_to_model', help='file containing the trained model.')  # or model_date_order.hdf5
    parser.add_argument('--dataset_name', type=str, default='tracings', help='name of the hdf5 dataset containing tracings')
    parser.add_argument('--output_file', default="./dnn_output.npy",  # or predictions_date_order.csv
                        help='output numpy file.')
    parser.add_argument('-bs', type=int, default=32, help='Batch size.')

    # Parse arguments, and warn about unknown arguments
    args, unk = parser.parse_known_args()
    if unk:
        warnings.warn("Unknown arguments:" + str(unk) + ".")

    # Import data using ECGSequence
    seq = ECGSequence(args.path_to_hdf5, args.dataset_name, batch_size=args.bs)

    # Import the pre-trained model
    model = load_model(args.path_to_model, compile=False)
    model.compile(loss='binary_crossentropy', optimizer=Adam())

    # Get model predictions on the test set
    y_score = model.predict(seq, verbose=1)

    # Save predictions as a numpy file
    np.save(args.output_file, y_score)

    print("Output predictions saved")
