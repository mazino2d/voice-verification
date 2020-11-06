from logger import Logger
import model
import dataset
import callbacks


if __name__ == "__main__":
    logger = Logger("main")
    generator_train_data = dataset.VoiceNpyDataGenerator("dataset/npy")

    logger.info("Model is starting ...")

    model_v1 = model.gen_model_v1()

    cp_callback = callbacks.get_check_point_callback()
    cp_tensorboard = callbacks.get_tensor_board_callback()

    model_v1.fit(
        generator_train_data,
        epochs=64,
        callbacks=[cp_callback, cp_tensorboard],
    )

    logger.info("Model is done!")
