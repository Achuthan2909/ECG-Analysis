# ECG-Analysis

## Multiple ECG Analysis Simulation

### Simulation of List of Papers:

1. **Ribeiro, A.H., Ribeiro, M.H., Paixão, G.M.M. et al.**
   - *Automatic diagnosis of the 12-lead ECG using a deep neural network.*
   - Published in [Nature Communications (2020)](https://www.nature.com/articles/s41467-020-15432-4).
   - DOI: [10.1038/s41467-020-15432-4](https://doi.org/10.1038/s41467-020-15432-4)

2. **ECG Signal Classification Using Deep Learning Techniques Based on the PTB-XL Dataset**
   - Published in [Entropy (August 2021)](https://www.mdpi.com/1099-4300/23/9/1121).
   - DOI: [10.3390/e23091121](https://doi.org/10.3390/e23091121)
   - License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
   - Implementation: First 2 models

### Dataset:

- The dataset used in this project is available on Zenodo. You can access it [here](https://zenodo.org/records/4916206).

### Scripts:

1. **Training Script (`train.py`):**
   - Run the training script with the following command:
     ```bash
     python train.py /path/to/hdf5_file /path/to/csv_file --val_split 0.2 --dataset_name tracings
     ```
   - Replace `/path/to/hdf5_file` and `/path/to/csv_file` with the actual paths to your HDF5 file and CSV file. Adjust the `--val_split` value and provide the `--dataset_name` as needed.

2. **Prediction Script (`predict.py`):**
   - Run the prediction script with the following command:
     ```bash
     python predict.py /path/to/hdf5_file /path/to/model_file --dataset_name tracings --output_file ./dnn_output.npy -bs 32
     ```
   - Replace `/path/to/hdf5_file` and `/path/to/model_file` with the actual paths to your HDF5 file and model file. Adjust other options such as `--dataset_name`, `--output_file`, and `-bs` as needed.

3. **Figures and Tables Script (`generate_figures_and_tables.py`):**
   - Run the figures and tables script with the following command:
     ```bash
     python generate_figures_and_tables.py /path/to/true_labels.csv /path/to/predicted_labels.npy --threshold 0.6
     ```
   - Replace `/path/to/true_labels.csv` and `/path/to/predicted_labels.npy` with the actual paths to your true labels CSV file and predicted labels NPY file. Adjust the threshold using `--threshold`.
