import os
import tensorflow.keras.callbacks as C


def get_check_point_callback(checkpoint_path="checkpoint"):
    os.makedirs(checkpoint_path, exist_ok=True)
    cp_callback = C.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True)

    return cp_callback