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
    def __init__(self, device, noise, rows, columns, hidden_size, lstm_layers=1, bidirectional=True):
        super(GANerAidGenerator, self).__init__()
        self.rows = rows
        self.columns = columns
        self.lstm_layers = lstm_layers
        self.hidden_size = hidden_size
        self.input_size = int(noise / columns)
        self.noise_size = noise
        self.directions = 2 if bidirectional else 1
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, lstm_layers, batch_first=True,
                            bidirectional=bidirectional)

        self.out = nn.Sequential(
            nn.Linear(int(self.hidden_size / self.rows) * self.columns, columns),
            nn.Tanh()
        )

        self.device = device

    def init_hidden(self, batch_size):
        # function to init the hidden layers
        return (torch.randn(self.lstm_layers * self.directions, batch_size, self.hidden_size).to(self.device),
                torch.randn(self.lstm_layers * self.directions, batch_size, self.hidden_size).to(self.device))

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
            output[:, c, :] = self.out(r).to(self.device)
            c += 1
        return output.to(self.device)


class GANerAidGAN:
    def __init__(self, noise, rows, columns, hidden_size, device, lstm_layers=1, dropout_d=0.3, leaky_relu_d=0.2,
                 bidirectional=True):
        super(GANerAidGAN, self).__init__()
        self.device = device
        self.rows = rows
        self.noise = noise
        self.columns = columns
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.dropout_d = dropout_d
        self.leaky_relu_d = leaky_relu_d
        self.bidirectional = bidirectional

        self.generator = GANerAidGenerator(device, noise, rows, columns, hidden_size, lstm_layers=self.lstm_layers,
                                           bidirectional=self.bidirectional).to(device)
        self.discriminator = GANerAidDiscriminator(rows, columns, dropout=self.dropout_d, leaky_relu=self.leaky_relu_d).to(device)

    def train(self):
        self.generator.train()
        self.discriminator.train()

    def eval(self):
        self.generator.eval()
        self.discriminator.eval()

    def get_params(self):
        return {"generator": self.generator.state_dict(),
                    "discriminator": self.discriminator.state_dict(),
                    "rows": self.rows,
                    "noise": self.noise,
                    "columns": self.columns,
                    "hidden_size": self.hidden_size,
                    "lstm_layers": self.lstm_layers,
                    "dropout_d": self.dropout_d,
                    "leaky_relu_d": self.leaky_relu_d,
                    "bidirectional": self.bidirectional
                    }

    @staticmethod
    def setup_from_params(params, device):
        generator_params = (params['generator'])
        discriminator_params = (params['discriminator'])

        rows = params["rows"]
        noise = params["noise"]
        columns = params["columns"]
        hidden_size = params["hidden_size"]
        lstm_layers = params["lstm_layers"]
        dropout_d = params["dropout_d"]
        leaky_relu_d = params["leaky_relu_d"]
        bidirectional = params["bidirectional"]

        gan = GANerAidGAN(noise, rows, columns, hidden_size, device, lstm_layers, dropout_d, leaky_relu_d, bidirectional)
        gan.generator.load_state_dict(generator_params)
        gan.discriminator.load_state_dict(discriminator_params)
        return gan
