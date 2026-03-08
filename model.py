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
    '''A simple convolutional block with two Conv2d layers, each followed by GroupNorm and SiLU activation.'''
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class MBConvBlock(nn.Module):
    '''
    Mobile inverted bottleneck block.
    expand → depthwise → project
    '''

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expand_ratio=4,
        groups=4,
        dilation: int = 1,
    ):
        super().__init__()

        mid_ch = in_ch * expand_ratio

        self.block = nn.Sequential(
            # expand
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.GroupNorm(groups, mid_ch),
            nn.SiLU(inplace=True),

            # depthwise
            nn.Conv2d(
                mid_ch,
                mid_ch,
                3,
                padding=dilation,
                dilation=dilation,
                groups=mid_ch,
                bias=False,
            ),
            nn.GroupNorm(groups, mid_ch),
            nn.SiLU(inplace=True),

            # project
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.GroupNorm(groups, out_ch)
        )

    def forward(self, x):
        return self.block(x)

class Up(nn.Module):
    '''An upsampling block that uses bilinear upsampling,
    concatenates with the matching skip connection, and refines features with ConvBlock.
    '''
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.reduce = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, bias=False)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)
        self.gate = GatedSkip(skip_ch, in_ch // 2, inter_ch=skip_ch // 2)

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
        self.context = ASPP(1280, 256)  # Context module at the bottleneck

    def forward(self, x):
        input_size = x.shape[2:]

        features = self.encoder(x)
        s1 = features['skip1']
        s2 = features['skip2']
        s3 = features['skip3']
        s4 = features['skip4']
        b = features['bottleneck']

        b = self.context(b)

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
    
class ASPP(nn.Module):

    def __init__(self, in_ch, out_ch, rates=(1,2,3)):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True)
        )

        self.branch2 = MBConvBlock(in_ch, out_ch, groups=8, dilation=rates[0])
        self.branch3 = MBConvBlock(in_ch, out_ch, groups=8, dilation=rates[1])
        self.branch4 = MBConvBlock(in_ch, out_ch, groups=8, dilation=rates[2])

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True)
        )

        concat_ch = out_ch * 5

        self.se = SEBlock(concat_ch)

        self.project = nn.Sequential(
            nn.Conv2d(concat_ch, in_ch, 1, bias=False),
            nn.GroupNorm(8, in_ch),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):

        h, w = x.shape[2:]

        p1 = self.branch1(x)
        p2 = self.branch2(x)
        p3 = self.branch3(x)
        p4 = self.branch4(x)

        p5 = self.pool(x)
        p5 = torch.nn.functional.interpolate(
            p5, size=(h, w), mode="bilinear", align_corners=False
        )

        x = torch.cat([p1,p2,p3,p4,p5], dim=1)

        x = self.se(x)

        return self.project(x)

class SEBlock(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        w = self.pool(x)
        w = self.fc(w)

        return x * w

class GatedSkip(nn.Module):

    def __init__(self, skip_ch, decoder_ch, inter_ch):
        super().__init__()

        self.skip_proj = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.dec_proj  = nn.Conv2d(decoder_ch, inter_ch, 1, bias=False)

        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, skip, decoder):

        g1 = self.skip_proj(skip)
        g2 = self.dec_proj(decoder)

        gate = nn.ReLU()(g1 + g2)
        gate = self.psi(gate)

        return skip * gate