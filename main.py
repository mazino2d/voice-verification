from logger import Logger
import model
import dataset

if __name__ == "__main__":
    logger = Logger("main")
    generator_train_data = dataset.VoiceNpyDataGenerator("dataset/npy")

    logger.info("Model is starting ...")
    model_v1 = model.gen_model_v1()
    model_v1.fit(generator_train_data, epochs=1)
    logger.info("Model is done!")
