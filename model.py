"""UNet model definition used for semantic segmentation."""

import torch
from torch import nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights, feature_extraction

backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1).features
return_nodes = {
        '1': 'skip1',
        '2': 'skip2',
        '3': 'skip3',
        '5': 'skip4',
        '7': 'bottleneck'
    }
encoder = feature_extraction.create_feature_extractor(backbone, return_nodes=return_nodes)

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
    '''An upsampling block that uses transposed convolution,
    concatenates with the matching skip connection, and refines features with ConvBlock.'''
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2, bias=False)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    '''A UNet architecture for semantic segmentation, consisting of an encoder, bottleneck, and decoder.'''
    def __init__(self, num_classes, backbone=encoder):
        super().__init__()

        # Encoder
        self.encoder = backbone

        # Decoder
        self.up1 = Up(1280, 160, 512)
        self.up2 = Up(512, 64, 256)
        self.up3 = Up(256, 48, 128)
        self.up4 = Up(128, 24, 64)

        self.head = nn.Conv2d(64, num_classes, 1)
        self.logits_up = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2)

    def forward(self, x):
        input_size = x.shape[2:]

        features = self.encoder(x)
        s1 = features['skip1']
        s2 = features['skip2']
        s3 = features['skip3']
        s4 = features['skip4']
        b = features['bottleneck']

        x = self.up1(b, s4)
        x = self.up2(x, s3)
        x = self.up3(x, s2)
        x = self.up4(x, s1)
        x = self.head(x)
        x = self.logits_up(x)

        # Guard against odd input sizes that can create off-by-one shape differences.
        if x.shape[2:] != input_size:
            x = x[:, :, :input_size[0], :input_size[1]]
        return x