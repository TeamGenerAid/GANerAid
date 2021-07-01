import numpy as np
import torch
from torch.autograd.variable import Variable


def set_or_default(key, default_value, args):
    return default_value if key not in args else args[key]


def noise(batch_size, noise_size):
    n = Variable(torch.randn(batch_size, noise_size))
    if torch.cuda.is_available(): return n.cuda()
    return n



def add_noise(x, noise_factor):
    if x < 0:
        return x + np.random.uniform(0, noise_factor)
    if x > 0:
        return x - np.random.uniform(0, noise_factor)

