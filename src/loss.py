import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

DELTA_K = 8
SIGMA = 1 / (12 * (np.log2(320 / 128)))


def e_t(y_t1, y_t2, sigma=SIGMA, delta_k=DELTA_K):
    return abs((y_t1 - y_t2) - sigma * delta_k)


def huber_loss(y_pred, clip_delta=0.25 * SIGMA):
    y_t1 = y_pred[0][0][-2]
    y_t2 = y_pred[0][0][-1]
    error = e_t(y_t1, y_t2)
    cond = abs(error) < clip_delta
    squared_loss = 0.5 * K.square(error)
    linear_loss = clip_delta * (abs(error) - 0.5 * clip_delta)
    return 100 * tf.where(cond, squared_loss, linear_loss)
