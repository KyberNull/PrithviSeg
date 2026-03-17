"""UNet model definition used for semantic segmentation."""

import torch
from torch import nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights, feature_extraction

# ===========================
# Encoder
# ===========================
backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1).features
return_nodes = {
        '1': 'skip1',
        '2': 'skip2',
        '3': 'skip3',
        '5': 'skip4',
        '7': 'bottleneck'
    }
encoder = feature_extraction.create_feature_extractor(backbone, return_nodes=return_nodes)

# ===========================
# Basic Blocks
# ===========================

class ConvBlock(nn.Module):
    """Two conv layers with GroupNorm + SiLU."""
    def __init__(self, in_ch, out_ch, groups=8, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    

# ===========================
# Gate for skip connections
# ===========================

class GatedSkip(nn.Module):

    def __init__(self, skip_ch, decoder_ch, inter_ch):
        super().__init__()

        self.relu = nn.ReLU()

        self.skip_proj = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.dec_proj  = nn.Conv2d(decoder_ch, inter_ch, 1, bias=False)

        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, skip, decoder):

        g1 = self.skip_proj(skip)
        g2 = self.dec_proj(decoder)

        gate = self.relu(g1 + g2)
        gate = self.psi(gate)

        return skip * gate

class Up(nn.Module):
    '''An upsampling block that uses bilinear upsampling,
    concatenates with the matching skip connection, and refines features with ConvBlock.
    '''
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.reduce = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, bias=False)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)
        self.gate = GatedSkip(skip_ch, in_ch // 2, in_ch // 2)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.reduce(x)
        if x.shape[2:] != skip.shape[2:]:
            x = torch.nn.functional.interpolate(
                x, size=skip.shape[2:], mode='bilinear', align_corners=False
            )
        skip = self.gate(skip, x)
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
        self.logits_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

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
