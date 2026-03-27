"""SegFormer model definition used for semantic segmentation."""

import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

# ===========================
# SegFormer Model
# ===========================

class SegFormer(nn.Module):
    """
    SegFormer architecture using a hierarchical Transformer encoder (MiT) 
     and a lightweight MLP decoder.
    """
    def __init__(self, num_classes: int, encoder_name: str = "mit_b2"):
        super().__init__()
        
        # We use smp.Segformer which implements the MiT backbone and MLP decoder.
        # mit_b2 is the recommended 'Quality' choice for RTX 50-series.
        self.model = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            classes=num_classes,
        )

    def forward(self, x):
        """
        Forward pass. 
        Note: SegFormer output is typically 1/4 of input resolution.
        SMP handles the upsampling to the original input size internally.
        """
        input_size = x.shape[2:]
        
        # Get logits from SegFormer
        logits = self.model(x)
        
        # Ensure output matches input size exactly (handles odd input dimensions)
        if logits.shape[2:] != input_size:
            logits = F.interpolate(
                logits, size=input_size, mode="bilinear", align_corners=False
            )
            
        return logits