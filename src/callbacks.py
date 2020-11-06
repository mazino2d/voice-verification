import os
import tensorflow.keras.callbacks as C


def get_check_point_callback(checkpoint_path="model/cp-{epoch:04d}.ckpt"):
    cp_callback = C.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        period=5,
    )

    return cp_callback


def get_tensor_board_callback(log_path="tensor-board"):
    cp_tensorboard = C.TensorBoard(log_path, histogram_freq=1)

    return cp_tensorboard