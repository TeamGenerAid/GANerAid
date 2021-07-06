# containing preprocessing logic
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
import numpy as np


def add_noise(value, binary_noise):
    if value < 0:
        return value + np.random.uniform(0, binary_noise)
    if value >= 0:
        return value - np.random.uniform(0, binary_noise)


def scale(x):
    if x < 0:
        return -1
    else:
        return 1


def get_binary_columns(data):
    binary_columns = []
    i = 0
    for column in data:
        if len(data[column].unique()) == 2:
            binary_columns.append(i)
        i = i + 1
    return binary_columns


class DataProcessor:
    def __init__(self, dataset):
        self.pandas_dataset = dataset
        self.binary_columns = get_binary_columns(self.pandas_dataset)
        self.sc = None

    def preprocess(self, binary_noise=0.2, aug_factor=0):
        np_data = self.pandas_dataset.to_numpy()
        self.sc = MinMaxScaler((-1, 1))
        self.sc = self.sc.fit(np_data)
        scaled_data = self.sc.fit_transform(np_data)

        # adding noise based on aug_factor
        if aug_factor > 0:
            copied_data = scaled_data.copy()

            for _ in range(aug_factor):
                for x in range(int(copied_data.shape[1])):
                    if x not in self.binary_columns:
                        for y in range(copied_data.shape[0]):
                            noise = np.random.uniform(.00001, .00001)
                            if -1 <= (copied_data[y, x] + noise) <= 1:
                                copied_data[y, x] = copied_data[y, x] + noise
                            else:
                                copied_data[y, x] = copied_data[y, x] - noise

            data2 = np.append(scaled_data, copied_data, axis=0)

            scaled_data = data2

        for i in self.binary_columns:
            scaled_data[:, i] = np.array([add_noise(x, binary_noise) for x in scaled_data[:, i]])

        return scaled_data

    def postprocess(self, data):
        # reverse binary
        for i in self.binary_columns:
            data[:, i] = np.array([scale(x) for x in data[:, i]])

        # reverse min max scaling
        data = pd.DataFrame(self.sc.inverse_transform(data))

        data.columns = self.pandas_dataset.columns

        for column in data.columns:
            if (self.pandas_dataset[column].dtype == "int64"):
                data[column] = data[column].round(0)

        data = data.astype(self.pandas_dataset.dtypes)

        return data
