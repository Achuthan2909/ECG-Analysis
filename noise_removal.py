import pybaselines
import numpy as np

class NR:
    def baselinewander(data):
    # Loop through batches and leads to apply ArPLS method
        for batch_idx in range(len(data)):
            batch_x, batch_y = data[batch_idx]
            for j in range(batch_x.shape[0]):
                reshaped_data = batch_x[j, :, :]
                for i in range(reshaped_data.shape[1]):
                    noisy_data = np.array(reshaped_data[:, i])
                    baseline, param1 = pybaselines.whittaker.arpls(noisy_data)
                    corrected_data = np.array(noisy_data) - np.array(baseline)
                    reshaped_data[:, i] = corrected_data
                # Update the batch data with corrected data
                batch_x[j, :, :] = reshaped_data
        return data
