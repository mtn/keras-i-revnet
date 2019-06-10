"Model utility functions"

import tensorflow as tf
from tensorflow.keras.layers import Permute, ZeroPadding2D
from math import ceil


def compute_block_size_shapes(axis_size, block_size):
    """
    Compute the split shapes, if block_size doesn't divide axis_size (tf.split
    only accepts a block_size that evenly divides axis_size)
    """
    assert axis_size >= block_size

    if axis_size % block_size == 0:
        return block_size
    else:
        num_blocks = ceil(axis_size / block_size)
        block_size_or_shapes = [block_size] * (num_blocks - 1)
        block_size_or_shapes.append(axis_size - (num_blocks - 1) * block_size)
        return block_size_or_shapes


def split(x):
    "Split along channel dimension"

    n = int(x.get_shape().as_list()[1] / 2)
    x1 = x[:, :n, :, :]
    x2 = x[:, n:, :, :]
    return x1, x2


def merge(x1, x2):
    "Merge along channel dimension"

    return tf.concat((x1, x2), axis=1)


class injective_pad(object):
    def __init__(self, pad_size):
        self.pad_size = pad_size
        self.pad = ZeroPadding2D(padding=((0, 0), (pad_size, 0)))

    def forward(self, x):
        x = Permute((2, 1, 3))(x)
        x = self.pad(x)
        return Permute((2, 1, 3))(x)

    def inverse(self, x):
        return x[:, : x.get_shape().as_list()[1] - self.pad_size, :, :]


class psi(object):
    def __init__(self, block_size):
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def inverse(self, x):
        output = Permute((1, 3, 4, 2))(x)
        (batch_size, d_height, d_width, d_depth) = output.size()

        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)

        t_1 = tf.reshape(
            output, (batch_size, d_height, d_width, self.block_size_sq, s_depth)
        )

        spl = tf.split(
            t_1, compute_block_size_shapes(t_1.shape[3], self.block_size), axis=3
        )
        stack = [
            tf.reshape(t_t, (batch_size, d_height, s_width, s_depth)) for t_t in spl
        ]

        # TODO transpose permutation?
        output = tf.reshape(
            Permute((1, 3, 2, 4, 5))(tf.transpose(tf.stack(stack, axis=0))),
            (batch_size, s_height, s_width, s_depth),
        )

        return Permute((1, 4, 2, 3))(output)

    def forward(self, inp):
        output = Permute((1, 3, 4, 2))(inp)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_height = int(s_height / self.block_size)

        t_1 = tf.split(
            output, compute_block_size_shapes(t_1.shape[2], self.block_size), axis=2
        )
        stack = [tf.reshape(t_t, (batch_size, d_height, d_depth)) for t_t in t_1]
        output = tf.stack(stack, axis=1)
        output = Permute((1, 3, 2, 4))(output)
        output = Permute((1, 4, 2, 3))(output)

        return output
