import os
import random as rd
import time as tm

import numpy as np
import tensorflow.keras.utils as U

from logger import Logger


class VoiceNpyDataGenerator(U.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        path_dataset: str,
        size_batch=8,
        num_batch=64,
        neg_per_pos=3,
        sample_rate=22050,
        num_seconds=5,
        num_channels=1,
    ):
        "Initialization"
        self.logger = Logger("dataset")
        self.logger.info("Data Generator is starting ...")
        # Bind default parameter
        self.path_dataset: str = path_dataset
        self.size_batch: int = size_batch
        self.num_batch: int = num_batch
        self.neg_per_pos: int = neg_per_pos
        self.num_pos: int = size_batch // (neg_per_pos + 1)
        self.num_neg: int = self.num_pos * neg_per_pos
        self.sample_rate: int = sample_rate
        self.num_seconds: int = num_seconds
        self.num_channels: int = num_channels
        self.input_shape = (sample_rate * num_seconds, num_channels)

        if self.size_batch % (self.neg_per_pos + 1) != 0:
            self.logger.warning("size_batch: %s" % (self.size_batch))
            self.logger.warning("neg_per_pos: %s" % (self.neg_per_pos))
            self.logger.warning("size_batch should be divisible by (neg_per_pos + 1)!")

        # Load ALL dataset
        self.ls_voices = []
        time_start = tm.time()
        for path_voice in os.listdir(self.path_dataset):
            voice = np.load("%s/%s/data.npy" % (self.path_dataset, path_voice))
            self.ls_voices.append(voice)
        self.num_voices = len(self.ls_voices)
        time_end = tm.time()

        # Log info
        self.logger.info("Load data form %s" % self.path_dataset)
        self.logger.info("Number of voices: %s" % self.num_voices)
        self.logger.info("Number of batchs: %s" % self.num_batch)
        self.logger.info("Batch size: %s" % self.size_batch)
        self.logger.info("Number of pos voice pair: %s" % self.num_pos)
        self.logger.info("Number of neg voice pair: %s" % self.num_neg)
        self.logger.info("Load time: %.2f (seconds)" % (time_end - time_start))

        self.logger.info("Data Generator is started!")

    def __len__(self):
        "Denotes the number of batches per epoch"
        return self.num_batch

    def __getitem__(self, index):
        "Generate one batch of data"
        X1 = np.empty((self.size_batch, *self.input_shape))
        X2 = np.empty((self.size_batch, *self.input_shape))
        Y = np.empty((self.size_batch), dtype=int)

        for idx in range(self.num_pos):
            choice_random = rd.randint(0, self.num_voices - 1)
            X1[idx] = self._get_random_sample(choice_random)
            X2[idx] = self._get_random_sample(choice_random)
            Y[idx] = 0

        for idx in range(self.num_pos, self.size_batch):
            ls_choices_random = rd.sample(range(0, self.num_voices), 2)
            X1[idx] = self._get_random_sample(ls_choices_random[0])
            X2[idx] = self._get_random_sample(ls_choices_random[1])
            Y[idx] = 1

        return [X1, X2], Y

    def _get_random_sample(self, idx):
        "Generate one ramdom subsample"
        size: int = self.ls_voices[idx].shape[0]
        start: int = rd.randint(0, size - self.input_shape[0])
        sample = self.ls_voices[idx][start : start + self.input_shape[0]]

        return np.expand_dims(sample, axis=1)


if __name__ == "__main__":
    try:
        dg = VoiceNpyDataGenerator("./dataset/npy", 8, 64)
        print(dg._get_random_sample(0))
        print(dg.__getitem__(0))
        input("Enter to continue -> ...")

    except Exception as e:
        Logger().error(msg=e)
