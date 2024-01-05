import pybaselines
import numpy as np

class NR:
    def baselinewander(data):
        # Loop through batches and leads to apply ArPLS method
        for batch_idx in range(len(data)):
            batch_x, batch_y = data[batch_idx]

            # Loop through samples in the batch
            for j in range(batch_x.shape[0]):
                reshaped_data = batch_x[j, :, :]

                # Loop through leads in the data
                for i in range(reshaped_data.shape[1]):
                    noisy_data = np.array(reshaped_data[:, i])

                    # Apply ArPLS method from pybaselines library
                    baseline, param1 = pybaselines.whittaker.arpls(noisy_data)

                    # Subtract baseline to obtain corrected data
                    corrected_data = np.array(noisy_data) - np.array(baseline)
                    reshaped_data[:, i] = corrected_data

                # Update the batch data with corrected data
                batch_x[j, :, :] = reshaped_data

        # Return the data with baseline-wander corrected samples
        return data

# Example Usage:
# corrected_data = NR.baselinewander(data)
