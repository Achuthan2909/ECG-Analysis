import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")
from keras.models import load_model
from keras.optimizers import Adam
from datasets import ECGSequence  # Assuming you have a module named 'datasets' with the ECGSequence class

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get performance on test set from hdf5')
    parser.add_argument('path_to_hdf5', type=str,
                        help='path to hdf5 file containing tracings')
    parser.add_argument('path_to_model',  # or model_date_order.hdf5
                        help='file containing training model.')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                        help='name of the hdf5 dataset containing tracings')
    parser.add_argument('--output_file', default="./dnn_output.npy",  # or predictions_date_order.csv
                        help='output csv file.')
    parser.add_argument('-bs', type=int, default=32,
                        help='Batch size.')

    args, unk = parser.parse_known_args()
    if unk:
        warnings.warn("Unknown arguments:" + str(unk) + ".")

    # Import data
    seq = ECGSequence(args.path_to_hdf5, args.dataset_name, batch_size=args.bs)
    
    # Import model2 (assuming model2 is in the same format as the original model)
    model2 = load_model('path_to_model2.hdf5', compile=False)
    model2.compile(loss='binary_crossentropy', optimizer=Adam())
    
    # Generate predictions using model2
    y_score_model2 = model2.predict(seq, verbose=1)

    # Save predictions for model2
    np.save(args.output_file.replace(".npy", "_model2.npy"), y_score_model2)

    print("Output predictions for model2 saved")
