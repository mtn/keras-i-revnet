"""
i-RevNet model implementation
"""

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, ReLU, AveragePooling2D


class iRevNet(tf.keras.Model):
    def __init__(
        self,
        nBlocks,
        nStrides,
        nClasses,
        nChannels=None,
        init_ds=2,
        dropout_rate=0.0,
        affineBN=True,
        in_shape=None,
        mult=4,
    ):
        super(iRevNet, self).__init__()
        self.ds = in_shape[2] // 2 ** (nStrides.count(2) + init_ds // 2)
        self.init_ds = init_ds
        self.in_ch = in_shape[0] * 2 ** self.init_ds
        self.nBlocks = nBlocks
        self.first = True

        print(f"\n == Building iRevNet {sum(nBlocks) * 3 + 1} ==")

        if not nChannels:
            nChannels = [
                self.in_ch // 2,
                self.in_ch // 2 * 4,
                self.in_ch // 2 * 4 ** 2,
                self.in_ch // 2 * 4 ** 3,
            ]

        self.init_psi = psi(self.init_ds)
        self.stack = self.irevnet_stack(
            irevnet_block,
            nChannels,
            nBlocks,
            nStrides,
            dropout_rate=droput_rate,
            affineBN=affineBN,
            in_ch=self.in_ch,
            mult=mult,
        )
        self.bn1 = BatchNormalization(axis=1, momentum=0.9)
        self.linear = Dense(nClasses)

    def call(self, x):
        """
        iRevNet forward pass. Returns the full network output and the output from the
        stack of invertible layers.
        """

        # Downsample using an invertible function psi
        if self.init_ds != 0:
            x = self.init_psi.forward(x)

        # Split on the channel dimension
        n = self.in_ch // 2
        out = (x[:, :n, :, :], x[:, n:, :, :])

        # Forward propagate through stack of invertible layers
        for block in self.stack:
            out = block(out)

        # This is the invertible, the operations which follow aren't
        stack_output = merge(out[0], out[1])

        out = ReLU(self.bn1(stack_output))
        out = AveragePooling2D(out, self.ds)
        out = tf.reshape(out, (out.shape[0], -1))
        out = self.linear(out)

        return out, stack_output

    def inverse(self, stack_output):
        """
        iRevNet inverse. `stack_output` is the output of the iRevNet stack.
        """

        # Compute inverses backwards through the iRevNet stack
        out = split(stack_output)
        for i in range(len(self.stack)):
            out = self.stack[-1 - i].inverse(out)
        out = merge(out[0], out[1])

        # Undo any downsampling
        if self.init_ds != 0:
            x = self.init_psi.inverse(out)
        else:
            x = out

        return x
