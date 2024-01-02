    
from keras.layers import Input, Conv1D, BatchNormalization, LeakyReLU, Flatten, Dense
from keras.models import Model
import numpy as np

class LeakyReLUUnit(object):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, x):
        return LeakyReLU(alpha=self.alpha)(x)

def get_custom_model():
    kernel_size = 16
    kernel_initializer = 'he_normal'
    signal = Input(shape=(4096, 12), dtype=np.float32, name='signal')
    x = signal

    # Five layers of one-dimensional convolutions with LeakyReLU activation
    for filters in [64, 128, 196, 256, 320]:
        x = Conv1D(filters, kernel_size, padding='same', use_bias=False,
                   kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization()(x)
        x = LeakyReLUUnit()(x)

    x = Flatten()(x)

    # Fully connected layer with sigmoid activation
    diagn = Dense(1, activation='sigmoid', kernel_initializer=kernel_initializer)(x)

    model = Model(signal, diagn)
    return model

if __name__ == "__main__":
    custom_model = get_custom_model()
    custom_model.summary()

