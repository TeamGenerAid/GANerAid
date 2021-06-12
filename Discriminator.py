import torch.nn as nn

class Discriminator(torch.nn.Module):

    def __init__(self, rows, columns, dropout=0.3, leaky_relu=0.2):
        super(Discriminator, self).__init__()
        in_features = rows * columns
        hidden_features_0 = int(in_features * 1.2)
        hidden_features_1 = int(in_features * 1)
        hidden_features_2 = int(in_features * 0.8)
        hidden_features_3 = int(in_features * 0.5)
        n_out = 1
        print("Init dense discriminator with hidden features {}".format(
            (hidden_features_0, hidden_features_1, hidden_features_2, hidden_features_3)))

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


class SimpleDenseDiscriminator(torch.nn.Module):

    def __init__(self, rows, columns, dropout=0.3, leaky_relu=0.2):
        super(SimpleDenseDiscriminator, self).__init__()
        in_features = rows * columns
        hidden_features_0 = 200
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(in_features, hidden_features_0),
            nn.LeakyReLU(leaky_relu),
            nn.Dropout(dropout)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(hidden_features_0, hidden_features_0),
            nn.LeakyReLU(leaky_relu),
            nn.Dropout(dropout)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(hidden_features_0, n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.out(x)
        return x
