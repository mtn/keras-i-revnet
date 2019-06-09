"Model utility functions"

import tensorflow as tf


def split(x):
    "Split along channel dimension"

    n = int(x.size()[1] / 2)
    x1 = x[:, :n, :, :]
    x2 = x[:, n:, :, :]
    return x1, x2


def merge(x1, x2):
    "Merge along channel dimension"

    return tf.concat((x1, x2), axis=1)
