"""
i-RevNet model implementation
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    ReLU,
    AveragePooling2D,
    Conv2D,
    Dropout,
)

from utils import psi, injective_pad, merge, split


class iRevNetBlock(tf.keras.layers.Layer):
    "i-RevNet block: block of reversible layers"

    def __init__(
        self,
        in_ch,
        out_ch,
        stride=1,
        first=False,
        dropout_rate=0.0,
        affineBN=True,
        mult=4,
    ):
        super(iRevNetBlock, self).__init__()

        self.pad = 2 * out_ch - in_ch
        self.stride = stride
        self.inj_pad = injective_pad(self.pad)
        self.psi = psi(stride)

        if self.pad != 0 and stride == 1:
            in_ch = out_ch * 2
            print("\n| Injective iRevNet |\n")

        layers = []
        if not first:
            layers.append(BatchNormalization(axis=1, center=affineBN, scale=affineBN))
            layers.append(ReLU())
        layers.append(
            Conv2D(
                int(out_ch // mult), 3, strides=stride, padding="same", use_bias=False, data_format="channels_first"
            )
        )
        # print("conv1 out channels", int(out_ch // mult))
        # exit()
        layers.append(BatchNormalization(axis=1, center=affineBN, scale=affineBN))
        layers.append(ReLU())
        layers.append(
            Conv2D(
                int(out_ch // mult), 3, strides=stride, padding="same", use_bias=False, data_format="channels_first"
            )
        )
        layers.append(Dropout(dropout_rate))
        layers.append(BatchNormalization(axis=1, center=affineBN, scale=affineBN))
        layers.append(ReLU())
        layers.append(Conv2D(out_ch, 3, strides=stride, padding="same", use_bias=False, data_format="channels_first"))

        self.bottleneck_block = layers

    def build(self, input_shape):
        "No custom trainable weights"
        super(iRevNetBlock, self).build(input_shape)

    def call(self, x):
        if self.pad != 0 and self.stride == 1:
            x = merge(x[0], x[1])
            x = self.inj_pad.forward(x)
            x1, x2 = split(x)
            x = (x1, x2)

        x1 = x[0]
        x2 = x[1]
        Fx2 = x2
        for block in self.bottleneck_block:
            Fx2 = block(Fx2)

        if self.stride == 2:
            x1 = self.psi.forward(x1)
            x2 = self.psi.forward(x2)

        y1 = Fx2 + x1

        return (x2, y1)

    def compute_output_shape(self, input_shape):
        pass

    def inverse(self, x):
        """ bijective or injecitve block inverse """
        x2, y1 = x[0], x[1]

        if self.stride == 2:
            x2 = self.psi.inverse(x2)

        Fx2 = x2
        for block in self.bottleneck_block:
            Fx2 = block(Fx2)
        Fx2 = -Fx2

        x1 = Fx2 + y1

        if self.stride == 2:
            x1 = self.psi.inverse(x1)

        if self.pad != 0 and self.stride == 1:
            x = merge(x1, x2)
            x = self.inj_pad.inverse(x)
            x1, x2 = split(x)
            x = (x1, x2)
        else:
            x = (x1, x2)

        return x


class iRevNet(tf.keras.Model):
    "i-RevNet model"

    def __init__(
        self,
        nBlocks,
        nStrides,
        nClasses,
        nChannels=None,
        init_ds=2,
        dropout_rate=0.,
        affineBN=True,
        in_shape=None,
        mult=4,
    ):
        "TODO document init params"
        super(iRevNet, self).__init__()

        self.ds = in_shape[2] // 2 ** (nStrides.count(2) + init_ds // 2)
        self.init_ds = init_ds
        self.in_ch = in_shape[0] * 2 ** self.init_ds
        self.nBlocks = nBlocks

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
            iRevNetBlock,
            nChannels,
            nBlocks,
            nStrides,
            dropout_rate=dropout_rate,
            affineBN=affineBN,
            in_ch=self.in_ch,
            mult=mult,
        )
        self.bn1 = BatchNormalization(axis=1, momentum=0.9)
        self.linear = Dense(nClasses)

    def irevnet_stack(
        self,
        block_constructor,
        nChannels,
        nBlocks,
        nStrides,
        dropout_rate,
        affineBN,
        in_ch,
        mult,
    ):
        "Create a stack of iRevNet blocks"

        blocks = []
        strides = []
        channels = []

        for channel, depth, stride in zip(nChannels, nBlocks, nStrides):
            strides = strides + ([stride] + [1] * (depth - 1))
            channels = channels + ([channel] * depth)

        for i, (channel, stride) in enumerate(zip(channels, strides)):
            blocks.append(
                block_constructor(
                    in_ch,
                    channel,
                    stride,
                    first=(i == 0),
                    dropout_rate=dropout_rate,
                    affineBN=affineBN,
                    mult=mult,
                )
            )
            in_ch = 2 * channel

        return blocks

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
        "iRevNet inverse. `stack_output` is the output of the iRevNet stack."

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


if __name__ == "__main__":
    model = iRevNet(
        nBlocks=[6, 16, 72, 6],
        nStrides=[2, 2, 2, 2],
        nChannels=None,
        nClasses=1000,
        init_ds=2,
        dropout_rate=0.0,
        affineBN=True,
        in_shape=[3, 224, 224],
        mult=4,
    )
