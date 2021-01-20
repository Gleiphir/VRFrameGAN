import torch
import torch.nn as nn
import torch.nn.functional as F
channels = 6
leak = 0.1
w_g = 4


class ZDepthAnalyzer(nn.Module):
    pass


class Generator(nn.Module):
    def __init__(self):
        """
        :arg
        in : 1x C x H x W,
            C = 5
        """
        super(Generator, self).__init__()
        #
        self.model = nn.Sequential(

            nn.Conv2d(5, 32, 3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, channels, 3, stride=1, padding=(1, 1)),
            nn.Sigmoid()
        )


    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(channels, 16, 3, stride=1, padding=(1, 1)),
            nn.LeakyReLU(leak),
            nn.Conv2d(16, 32, 4, stride=2, padding=(1, 1)),
            nn.LeakyReLU(leak),
            #nn.Conv2d(32, 32, 3, stride=1, padding=(1, 1)),
            #nn.LeakyReLU(leak),
            nn.Conv2d(32, 64, 4, stride=2, padding=(1, 1)),
            nn.LeakyReLU(leak),
            #nn.Conv2d(64, 64, 3, stride=1, padding=(1, 1)),
            #nn.LeakyReLU(leak),
        )
        self.fc = nn.Linear(128 * 128 * 5 * 4, 1) #327680 = 2^16 * 5
        self.sig = nn.Sigmoid()

    def forward(self, x):
        m = x
        m = self.seq(m)
        return self.sig(self.fc(m.view(-1, 128 * 128 * 5 * 4)))


