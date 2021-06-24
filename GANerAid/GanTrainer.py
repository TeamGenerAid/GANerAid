# contains logic for training the gan
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np

def noise(batch_size, noise_size):
    n = Variable(torch.randn(batch_size, noise_size))
    if torch.cuda.is_available(): return n.cuda()
    return n

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
        #optimizer
        self.weight_decay_d = 0
        self.weight_decay_g = 0.00001
        self.loss  = nn.BCELoss()

        self.verbose = verbose



    def train(self, dataset, gan, epochs=100):
        data_loader = torch.utils.data.DataLoader(RealData(dataset, rows=25), batch_size=100, shuffle=True)
        self.d_optimizer = optim.Adam(gan.discriminator.parameters(), lr=self.lr_d, weight_decay=self.weight_decay_d)
        self.g_optimizer = optim.Adam(gan.generator.parameters(), lr=self.lr_g, weight_decay=self.weight_decay_g)

        for epoch in range(epochs):
            gan.train()
            for n_batch, real_batch, in enumerate(data_loader):
                # 1. Train Discriminator
                real_data = Variable(real_batch)
                if torch.cuda.is_available(): real_data = real_data.cuda()
                # Generate fake data
                fake_data = gan.generator(noise(real_data.size(0))).detach()
                # Train D
                # Reset gradients
                self.d_optimizer.zero_grad()

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
                self.d_optimizer.step()

                # 2. Train Generator
                # Generate fake data
                fake_data = gan.generator(noise(real_batch.size(0)))
                # Train G
                # Reset gradients
                self.g_optimizer.zero_grad()
                # Sample noise and generate fake data
                prediction = gan.discriminator(fake_data)
                # Calculate error and backpropagate
                error = self.loss(prediction, real_data_target(prediction.size(0)))
                error.backward()
                # Update weights with gradients
                self.g_optimizer.step()


class RealData(Dataset):

    def __init__(self, dataset, rows=50):

        self.rows = rows
        self.dataset = dataset

    def sample_real_data(self, n):
        # indices = np.random.choice(scaled_data.shape[0], n, replace=False)
        output = np.empty([n, self.dataset.shape[1]])
        for i in range(n):
            rnd = np.random.randint(self.dataset.shape[0])
            output[i] = self.dataset[rnd]
        return output

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return Variable(torch.tensor(self.sample_real_data(self.rows)).float())




