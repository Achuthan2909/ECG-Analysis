from keras.layers import Input, Conv1D, LeakyReLU, AveragePooling1D, Flatten, Dense, Concatenate, Dropout
from keras.models import Model
import numpy as np
from scipy.stats import entropy
from pyentrp import entropy as ent

def convolutional_block(input_signal, num_layers=5):
    # Convolutional block
    x = input_signal
    for i in range(num_layers):
        x = Conv1D(64, kernel_size=3, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        
        # Residual connection with original signal
        residual_connection = AveragePooling1D(pool_size=2, strides=2, padding='same')(input_signal)
        x = Concatenate()([x, residual_connection])

    return x

def entropy_calculation(input_signal):
    num_channels = input_signal.shape[1]
    entropy_vector = np.zeros((num_channels, 9))  # Initialize an array for 9 entropy methods

    for i in range(num_channels):
        # 1. Shannon Entropy
        shannon_entropy = entropy(input_signal[:, i])

        # 2. Approximate Entropy
        approx_entropy = ent.ap_entropy(input_signal[:, i], order=2, metric='chebyshev')

        # 3. Sample Entropy
        sample_entropy = ent.sample_entropy(input_signal[:, i])

        # 4. Permutation Entropy
        permutation_entropy = ent.perm_entropy(input_signal[:, i], order=3, normalize=True)

        # 5. Spectral Entropy
        spectral_entropy = ent.spectral_entropy(input_signal[:, i], sf=1000, method='welch', normalize=True)

        # 6. SVD Entropy
        # Note: This is a simplified example; actual implementation depends on the specifics of SVD entropy
        svd_entropy = ent.svd_entropy(input_signal[:, i], order=3, delay=1, normalize=True)

        # 7. Rényi Entropy
        renyi_entropy = ent.renyi_entropy(input_signal[:, i], order=2)

        # 8. Tsallis Entropy
        tsallis_entropy = ent.tsallis_entropy(input_signal[:, i], q=2)

        # 9. Extropy
        extropy = entropy(np.abs(input_signal[:, i]))

        # Store the computed entropies in the vector
        entropy_vector[i, :] = [shannon_entropy, approx_entropy, sample_entropy,
                                permutation_entropy, spectral_entropy, svd_entropy, renyi_entropy,
                                tsallis_entropy, extropy]

    return entropy_vector
def fully_connected_block(input_features, n_classes=2):
    # Fully connected layers
    x = Flatten()(input_features)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.2)(x)  # Dropout for regularization
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.2)(x)  # Dropout for regularization

    # Output layer with softmax activation for binary classification
    output = Dense(n_classes, activation='softmax')(x)

    return output

def build_model(input_shape=(4096, 12), n_classes=2):
    # Input layer for the raw ECG signal
    input_signal = Input(shape=input_shape, name='input_signal')

    # Convolutional block
    conv_block_output = convolutional_block(input_signal)

    # Entropy block
    entropy_block_output = entropy_calculation(input_signal)

    # Concatenate outputs of both blocks
    concatenated_features = Concatenate(axis=-1)([conv_block_output, entropy_block_output])

    # Fully connected block
    output = fully_connected_block(concatenated_features, n_classes)

    # Create and compile the model
    model = Model(inputs=input_signal, outputs=output)
    return model

# Example usage:
if __name__ == "__main__":
    model = build_model()
    model.summary()
