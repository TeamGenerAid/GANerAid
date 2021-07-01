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







