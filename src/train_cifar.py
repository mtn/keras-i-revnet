"Train i-RevNet on CIFAR-10"

from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import argparse

from model import iRevNet


def make_parser():
    parser = argparse.ArgumentParser(description="Train i-RevNet/RevNet on Cifar")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--batch", default=128, type=int, help="batch size")
    parser.add_argument("--init_ds", default=0, type=int, help="initial downsampling")
    parser.add_argument("--epochs", default=200, type=int, help="number of epochs")
    parser.add_argument("--nBlocks", nargs="+", type=int)
    parser.add_argument("--nStrides", nargs="+", type=int)
    parser.add_argument("--nChannels", nargs="+", type=int)
    parser.add_argument(
        "--bottleneck_mult", default=4, type=int, help="bottleneck multiplier"
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--dataset", default="cifar10", type=str, help="dataset (cifar-10 or cifar-100)"
    )

    return parser


def train(model, train_data, datagen, epochs=50, batch_size=32):
    model, save_name = model
    x_train, y_train = train_data

    optimizer = tf.keras.optimizers.SGD(lr=1e-3)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    for e in range(epochs):
        print(f"Epoch {e}")
        batches = 0
        for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=batch_size):
            with tf.GradientTape() as tape:
                logits = model(x_batch)
                loss_value = loss_fn(y_batch, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            print(f"Loss = {float(loss_value)}")

            batches += 1
            if batches > len(x_train) // batch_size:
                break


def main():
    parser = make_parser()
    args = parser.parse_args()

    in_shape = [3, 32, 32]
    if args.dataset.lower() == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        nClasses = 10
    else:
        assert (
            args.dataset.lower() == "cifar100"
        ), "Only cifar10 and cifar100 are supported"
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        nClasses = 100

    # Switch from channel_last to channel_first
    x_train = np.moveaxis(x_train, 3, 1)
    x_test = np.moveaxis(x_test, 3, 1)
    assert x_train.shape[1] == 3, x_train.shape
    assert x_test.shape[1] == 3, x_test.shape

    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Some pre-processing differences from the original
    # TODO fit it on the data to do featurewise centering, and normalization
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        featurewise_center=True,
        featurewise_std_normalization=True,
        data_format="channels_first",
    )

    def get_model(args):
        model = iRevNet(
            nBlocks=args.nBlocks,
            nStrides=args.nStrides,
            nChannels=args.nChannels,
            nClasses=nClasses,
            init_ds=args.init_ds,
            dropout_rate=0.1,
            affineBN=True,
            in_shape=in_shape,
            mult=args.bottleneck_mult,
        )
        fname = "i-revnet-" + str(sum(args.nBlocks) + 1)
        return model, fname

    model, fname = get_model(args)

    train(
        (model, fname),
        (x_train, y_train),
        datagen,
        epochs=args.epochs,
        batch_size=args.batch,
    )


if __name__ == "__main__":
    tf.enable_eager_execution()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    main()
