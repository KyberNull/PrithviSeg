import torch
from torch import nn

class UNet(nn.Module):
    def __init__(self, num_class):
        super(UNet, self).__init__()

        self.Encoder1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, padding=3),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=7, padding=3),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
        )
        
        self.Encoder2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=7, padding=3),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=7, padding=3),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
        )

        self.Encoder3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=7, padding=3),
            nn.GroupNorm(4, 128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=7, padding=3),
            nn.GroupNorm(4, 128),
            nn.ReLU(),
        )

        self.Encoder4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=7, padding=3),
            nn.GroupNorm(4, 256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=7, padding=3),
            nn.GroupNorm(4, 256),
            nn.ReLU(),
        )

        self.Encoder5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=7, padding=3),
            nn.GroupNorm(4, 512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=7, padding=3),
            nn.GroupNorm(4, 512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
        )

        self.Decoder1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=7, padding=3),
            nn.GroupNorm(4, 256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=7, padding=3),
            nn.GroupNorm(4, 256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
        )

        self.Decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=7, padding=3),
            nn.GroupNorm(4, 128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=7, padding=3),
            nn.GroupNorm(4, 128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        )

        self.Decoder3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=7, padding=3),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=7, padding=3),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
        )

        self.Decoder4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=7, padding=3),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=7, padding=3),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.Conv2d(32, num_class, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.Encoder1(x)
        x2 = self.Encoder2(x1)
        x3 = self.Encoder3(x2)
        x4 = self.Encoder4(x3)
        x = self.Encoder5(x4)
        x = torch.cat([x, x4], dim=1)
        x = self.Decoder1(x)
        x = torch.cat([x, x3], dim=1)
        x = self.Decoder2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.Decoder3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.Decoder4(x)
        return x