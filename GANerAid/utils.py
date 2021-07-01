import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.autograd.variable import Variable
import torch


def set_or_default(key, default_value, args):
    return default_value if key not in args else args[key]

def noise(batch_size, noise_size):
    n = Variable(torch.randn(batch_size, noise_size))
    if torch.cuda.is_available(): return n.cuda()
    return n


def get_binary_columns(data):
    binary_columns = []
    i = 0
    for column in data:
        if len(data[column].unique()) == 2:
            binary_columns.append(i)
        i = i + 1
    return binary_columns



def add_noise(x, noise_factor):
    if x < 0:
        return x + np.random.uniform(0, noise_factor)
    if x > 0:
        return x - np.random.uniform(0, noise_factor)


def scale_and_add_noise(data, noise_factor, binary_columns):
    numpy_data = data.to_numpy()
    sc = MinMaxScaler((-1, 1))
    sc = sc.fit(numpy_data)
    scaled_data = sc.fit_transform(numpy_data)

    noise_factor = 0.2

    for i in binary_columns:
        scaled_data[:, i] = np.array([add_noise(x, noise_factor) for x in scaled_data[:, i]])
    return scaled_data
