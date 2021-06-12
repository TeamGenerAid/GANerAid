import torch.nn as nn

class DenseGenerator(torch.nn.Module):
    def __init__(self, noise, rows, columns, dropout=0.3, leaky_relu=0.2):
        super(DenseGenerator, self).__init__()
        in_features = noise
        hidden_features_0 = int(noise * 2)
        hidden_features_1 = int(noise * 4)
        hidden_features_2 = int(noise * 6)
        hidden_features_3 = int(noise * 8)
        n_out = rows * columns
        print("Init dense generator with hidden features {}".format(
            (hidden_features_0, hidden_features_1, hidden_features_2, hidden_features_3)))

        self.rows = rows
        self.columns = columns

        self.hidden0 = nn.Sequential(
            nn.Linear(in_features, hidden_features_0),
            # nn.Dropout(dropout),
            nn.LeakyReLU(leaky_relu)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(hidden_features_0, hidden_features_1),
            # nn.Dropout(dropout),
            nn.LeakyReLU(leaky_relu)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(hidden_features_1, hidden_features_2),
            # nn.Dropout(dropout),
            nn.LeakyReLU(leaky_relu)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(hidden_features_2, hidden_features_3),
            # nn.Dropout(dropout),
            nn.LeakyReLU(leaky_relu)
        )

        self.out = nn.Sequential(
            nn.Linear(hidden_features_3, n_out),
            # nn.Dropout(dropout),
            nn.Tanh()
        )

    def forward(self, x):
        # flatten the input for the dense layers
        x = torch.flatten(x, 1)
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.out(x)
        # reshape into input data
        x = x.view(x.shape[0], self.rows, self.columns)
        return x


class LstmGeneratorV3(nn.Module):
    def __init__(self, noise, rows, columns, hidden_size, lstm_layers=1, leaky_relu=0.2, bidirectional=True):
        super(LstmGeneratorV3, self).__init__()
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
        print(
            "Init lstm generator with hidden features {} and nr of layers {} and hidden dense layer with size {}".format(
                self.hidden_size, self.lstm_layers, self.hidden_size * columns * self.directions))

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

    def init_hidden(self, batch_size):
        # function to init the hidden layers
        return (torch.randn(self.lstm_layers * self.directions, batch_size, self.hidden_size).cuda(),
                torch.randn(self.lstm_layers * self.directions, batch_size, self.hidden_size).cuda())


