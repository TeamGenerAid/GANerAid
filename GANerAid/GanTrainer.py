# contains logic for training the gan
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import trange
from UtilityFunctions import noise


def real_data_target(batch_size):
    data = Variable(torch.ones(batch_size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data


def fake_data_target(batch_size):
    data = Variable(torch.zeros(batch_size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data


class GanTrainer:
    def __init__(self, lr_d, lr_g, verbose=True):
        self.lr_d = lr_d
        self.lr_g = lr_g
        # optimizer
        self.weight_decay_d = 0
        self.weight_decay_g = 0.00001
        self.loss = nn.BCELoss()

        self.verbose = verbose

    def train(self, dataset, gan, epochs=100, batch_size=100):
        data_loader = torch.utils.data.DataLoader(RealData(dataset, rows=gan.generator.rows), batch_size=batch_size,
                                                  shuffle=True)
        d_optimizer = optim.Adam(gan.discriminator.parameters(), lr=self.lr_d, weight_decay=self.weight_decay_d)
        g_optimizer = optim.Adam(gan.generator.parameters(), lr=self.lr_g, weight_decay=self.weight_decay_g)

        with trange(epochs) as tr:
            for it in tr:
                gan.train()
                for n_batch, real_batch, in enumerate(data_loader):
                    # 1. Train Discriminator
                    real_data = Variable(real_batch)
                    if torch.cuda.is_available(): real_data = real_data.cuda()
                    # Generate fake data
                    fake_data = gan.generator(noise(real_data.size(0), gan.generator.noise_size)).detach()
                    # Train D
                    # Reset gradients
                    d_optimizer.zero_grad()

                    # 1.1 Train on Real Data
                    prediction_real = gan.discriminator(real_data)
                    # Calculate error and backpropagate
                    error_real = self.loss(prediction_real, real_data_target(real_data.size(0)))
                    error_real.backward()

                    # 1.2 Train on Fake Data
                    prediction_fake = gan.discriminator(fake_data)
                    # Calculate error and backpropagate
                    error_fake = self.loss(prediction_fake, fake_data_target(real_data.size(0)))
                    error_fake.backward()

                    # 1.3 Update weights with gradients
                    d_optimizer.step()

                    # 2. Train Generator
                    # Generate fake data
                    fake_data = gan.generator(noise(real_batch.size(0), gan.generator.noise_size))
                    # Train G
                    # Reset gradients
                    g_optimizer.zero_grad()
                    # Sample noise and generate fake data
                    prediction = gan.discriminator(fake_data)
                    # Calculate error and backpropagate
                    g_error = self.loss(prediction, real_data_target(prediction.size(0)))
                    g_error.backward()
                    # Update weights with gradients
                    g_optimizer.step()
                    tr.set_postfix(loss="d error: {} --- g error {}".format(error_real + error_fake, g_error))


class RealData(Dataset):

    def __init__(self, dataset, rows=50):
        self.rows = rows
        self.dataset = dataset
        self.indices = [x for x in range(dataset.shape[0])]
        self.len = int(self.dataset.shape[0] / self.rows)

    def sample_real_data(self, n):
        output = np.empty([n, self.dataset.shape[1]])
        for i in range(n):
            rnd = np.random.randint(low=0, high=len(self.indices))
            idx = self.indices.pop(rnd)
            output[i] = self.dataset[idx]
        #reset indices
        if len(self.indices) == 0:
            self.indices = [x for x in range(self.dataset.shape[0])]
        return output

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return Variable(torch.tensor(self.sample_real_data(self.rows)).float())
