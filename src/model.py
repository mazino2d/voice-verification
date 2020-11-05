from tensorflow.keras import Model
import kapre as kp
import tensorflow.keras.layers as La
import tensorflow.keras.losses as Lo
import tensorflow.keras.optimizers as O
import tensorflow.keras.metrics as M

import os

import config, loss

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def concat_layer(
    inpt1: La.Layer,
    inpt2: La.Layer,
) -> La.Layer:
    branch1 = La.Reshape((1, *inpt1.shape[1:]), name="reshape_branch_1")(inpt1)
    branch2 = La.Reshape((1, *inpt2.shape[1:]), name="reshape_branch_2")(inpt2)
    oupt = La.Concatenate(axis=1, name="concat_1")([branch1, branch2])

    return oupt


def flatten(inpt: La.Layer, idx: int) -> La.Layer:
    oupt = La.TimeDistributed(La.Flatten(), name="td_flat_%s" % idx)(inpt)

    return oupt


def gen_stft(
    inpt: La.Layer,
    idx: int,
) -> La.Layer:
    oupt: La.Layer = kp.STFT(
        n_fft=2048,
        win_length=2048,
        hop_length=1024,
        pad_end=False,
        name="stft_%s" % idx,
        input_data_format="channels_last",
        output_data_format="channels_last",
    )(inpt)
    oupt = kp.Magnitude(name="mag_%s" % idx)(oupt)
    oupt = kp.MagnitudeToDecibel(name="mag2dec_%s" % idx)(oupt)

    return oupt


def cnn_2d(
    inpt: La.Layer,
    idx: int,
    num_node=8,
    kernel_size=(3, 3),
    pool_size=(2, 2),
    strides=(1, 1),
) -> La.Layer:
    # 2D convolution layer
    conv = La.Conv2D(num_node, kernel_size, strides, padding="same", activation="relu")
    oupt = La.TimeDistributed(conv, name="conv_%s" % (idx))(inpt)
    # Batch normalization layer
    oupt = La.BatchNormalization(name="batch_%s" % idx)(oupt)
    # Max pooling operation for 2D spatial data.
    pool = La.MaxPooling2D(pool_size)
    oupt = La.TimeDistributed(pool, name="max_pool_%s" % idx)(oupt)

    return oupt


def sigmoid(inpt: La.Layer) -> La.Layer:
    oupt = La.TimeDistributed(La.Dense(8, activation="relu"), name="relu")(inpt)
    oupt = La.TimeDistributed(La.Dense(1, activation="sigmoid"), name="sigmoid")(oupt)

    return oupt


def reshape(inpt: La.Layer, idx: int, shape) -> La.Layer:
    oupt = La.Reshape(shape, name="reshape_%s" % idx)(inpt)

    return oupt


def gen_model_v1() -> Model:
    inpt1 = La.Input(shape=config.INPUT_SHAPE, name="input_1")
    inpt2 = La.Input(shape=config.INPUT_SHAPE, name="input_2")

    oupt = concat_layer(gen_stft(inpt=inpt1, idx=1), gen_stft(inpt=inpt2, idx=2))

    oupt = cnn_2d(oupt, 1, 16, (3, 3), (1, 3), (2, 2))
    oupt = cnn_2d(oupt, 2, 32, (3, 3), (1, 3), (2, 2))
    oupt = cnn_2d(oupt, 3, 64, (3, 3), (3, 3), (2, 2))
    oupt = cnn_2d(oupt, 4, 64, (3, 3), (2, 2), (2, 2))

    oupt = flatten(oupt, 1)

    oupt = sigmoid(oupt)

    oupt = reshape(oupt, 1, (1, 2))

    oupt = La.Lambda(lambda y: loss.huber_loss(y), output_shape=(1,), name="loss")(oupt)

    model = Model(inputs=[inpt1, inpt2], outputs=[oupt])

    model.compile(optimizer=O.Adam(config.LR_ADAM), loss=Lo.Huber())

    return model


if __name__ == "__main__":
    model_v1 = gen_model_v1()
    model_v1.summary()