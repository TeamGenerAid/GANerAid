# containing preprocessing logic

from UtilityFunctions import get_binary_columns
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def add_noise(x, binary_noise):
    if x < 0:
        return x + np.random.uniform(0, binary_noise)
    if x > 0:
        return x - np.random.uniform(0, binary_noise)


class Preprocessing:
    def __init__(self, dataset):
        self.pandas_dataset = dataset
        self.binary_columns = get_binary_columns(self.pandas_dataset)
        self.sc = None

    def preprocess(self, binary_noise=0.2):
        np_data = self.pandas_dataset.to_numpy()
        self.sc = MinMaxScaler((-1, 1))
        self.sc = self.sc.fit(np_data)
        scaled_data = self.sc.fit_transform(np_data)

        for i in self.binary_columns:
            scaled_data[:, i] = np.array([add_noise(x, binary_noise) for x in scaled_data[:, i]])
        return scaled_data
