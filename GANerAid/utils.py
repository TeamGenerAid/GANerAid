import pandas as pd
import seaborn as sns
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



def read_CSV(path):
    data = pd.read_csv(path)
    return data


def read_excel(path):
    data = pd.read_excel(path)
    return data


def get_binary_columns(data):
    binary_columns = []
    i = 0
    for column in data:
        if len(data[column].unique()) == 2:
            binary_columns.append(i)
        i = i + 1
    return binary_columns


def plot_correlation(data):
    # TODO: make correlation matrix look pretty
    corr = data.corr()
    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)


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


def generate_data(sample_size, data):
    # TODO: generate Data
    return data


def evaluate(real_data, generated_data):
    # TODO: evaluation template
    plot_correlation(real_data)
    plot_correlation(generated_data)
    plot_correlation(real_data - generated_data)
    return
