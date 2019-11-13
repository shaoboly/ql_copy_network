import tensorflow as tf
import numpy as np


def guass_noisy(x,scale=0.1):
    x_corrupt = tf.add(x,scale*tf.random_normal(shape=tf.shape(x)))
    return x_corrupt

def mask_prob(x, keep_prob=0.9):
    x_corrupt = tf.nn.dropout(x, keep_prob)
    return x_corrupt



# numpy
def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval = low, maxval = high, dtype = tf.float32)



def salt_and_pepper_noise(X,v):
    X_noise = X.copy()
    n_features = X.shape[1]
    mn = X.min()
    mx = X.max()
    for i, sample in enumerate(X):
        mask = np.random.randint(0, n_features, v)
        for m in mask:
            if np.random.rand() < .5:
                X_noise[i][m] = mn
            else:
                X_noise[i][m] = mx
    return X_noise