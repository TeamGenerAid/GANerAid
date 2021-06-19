# containing the gan

from torch import nn, optim
import torch


# Discriminator
class GANerAidDiscriminator(torch.nn.Module):

    def __init__(self, rows, columns, dropout=0.3, leaky_relu=0.2):
        super(GANerAidDiscriminator, self).__init__()
        in_features = rows * columns
        hidden_features_0 = int(in_features * 1.2)
        hidden_features_1 = int(in_features * 1)
        hidden_features_2 = int(in_features * 0.8)
        hidden_features_3 = int(in_features * 0.5)
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(in_features, hidden_features_0),
            nn.LeakyReLU(leaky_relu),
            nn.Dropout(dropout)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(hidden_features_0, hidden_features_1),
            nn.LeakyReLU(leaky_relu),
            nn.Dropout(dropout)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(hidden_features_1, hidden_features_2),
            nn.LeakyReLU(leaky_relu),
            nn.Dropout(dropout)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(hidden_features_2, hidden_features_3),
            nn.LeakyReLU(leaky_relu),
            nn.Dropout(dropout)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(hidden_features_3, n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.out(x)
        return x


class GANerAidGenerator(nn.Module):
    def __init__(self, noise, rows, columns, hidden_size, lstm_layers=1, leaky_relu=0.2, bidirectional=True):
        super(GANerAidGenerator, self).__init__()
        self.rows = rows
        self.columns = columns
        self.lstm_layers = lstm_layers
        self.hidden_size = hidden_size
        self.input_size = int(noise / columns)
        self.directions = 2 if bidirectional else 1
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, lstm_layers, batch_first=True,
                            bidirectional=bidirectional)

        self.out = nn.Sequential(
            nn.Linear(int(self.hidden_size / self.rows) * self.columns, columns),
            nn.Tanh()
        )

    def init_hidden(self, batch_size):
        # function to init the hidden layers
        return (torch.randn(self.lstm_layers * self.directions, batch_size, self.hidden_size).cuda(),
                torch.randn(self.lstm_layers * self.directions, batch_size, self.hidden_size).cuda())

        # create a forward pass function'

    def forward(self, x):
        # reshape noise into lstm seq
        x = x.view(x.shape[0], self.columns, -1)
        hidden = self.init_hidden(x.shape[0])
        x, hidden = self.lstm(x, hidden)
        # flatten input for dense layers
        output = torch.zeros(x.shape[0], self.rows, self.columns)
        c = 0
        step = int(self.hidden_size / self.rows)
        for i in range(0, self.hidden_size, step):
            r = torch.flatten(x[:, :, i:step + i], 1)
            output[:, c, :] = self.out(r).cuda()
            c += 1
        return output.cuda()


class GANerAidGAN(nn.Module):
    def __init__(self, noise, rows, columns, hidden_size, device, lstm_layers=1, droput_d=0.3, leaky_relu_d=0.2,
                 leaky_relu_g=0.2, bidirectional=True):
        super(GANerAidGAN, self).__init__()
        self.generator = GANerAidGenerator(noise, rows, columns, hidden_size, lstm_layers=1, leaky_relu=0.2,
                                           bidirectional=True).to(device)
        self.discriminator = GANerAidDiscriminator(rows, columns, dropout=0.3, leaky_relu=0.2).to(device)

    def train(self):
        self.generator.train()
        self.disriminator.train()

    def eval(self):
        self.generator.eval()
        self.disriminator.eval()

    def save(self, path):
        torch.save(self.generator, path + "_generator")
        torch.save(self.discriminator, path + "discriminator")

    @staticmethod
    def load(self, path, device):
        generator = torch.load(path + "_generator")
        discriminator = torch.load(+ "discriminator")
        return self(generator, discriminator)









