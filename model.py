"""UNet model definition used for semantic segmentation."""

import torch
from torch import nn

class ConvBlock(nn.Module):
    '''
    A convolutional block consisting of two convolutional layers, each followed by group normalization and ReLU activation.
    This enhances representation while having less computation than a bigger kernel size.
    '''
    def __init__(self, in_ch: int, out_ch: int, groups=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.gn1(self.conv1(x)))
        x = self.act(self.gn2(self.conv2(x)))
        return x

class Up(nn.Module):
    '''An upsampling block that applies bilinear interpolation to upsample the input,
    concatenates it with the corresponding skip connection, and applies a convolutional block.'''
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.proj = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, bias=False)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.proj(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class Down(nn.Module):
    '''A downsampling block that applies max pooling followed by a convolutional block.'''
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))

class UNet(nn.Module):
    '''A UNet architecture for semantic segmentation, consisting of an encoder, bottleneck, and decoder.'''
    def __init__(self, num_classes):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(3, 32)
        self.enc2 = Down(32, 64)
        self.enc3 = Down(64, 128)
        self.enc4 = Down(128, 256)

        # Bottleneck
        self.bottleneck = Down(256, 512)

        # Decoder
        self.up1 = Up(512, 256, 256)
        self.up2 = Up(256, 128, 128)
        self.up3 = Up(128, 64, 64)
        self.up4 = Up(64, 32, 32)

        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        b = self.bottleneck(s4)

        x = self.up1(b, s4)
        x = self.up2(x, s3)
        x = self.up3(x, s2)
        x = self.up4(x, s1)

        return self.head(x)