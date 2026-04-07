"""SegFormer model definition used for semantic segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch.utils.checkpoint import checkpoint

# ===========================
# SegFormer Model
# ===========================

class SegFormer(nn.Module):
    """
    SegFormer architecture using a hierarchical Transformer encoder (MiT) 
     and a lightweight MLP decoder.
    """
    def __init__(self, num_classes: int, encoder_name: str = "mit_b2", use_gradient_checkpointing: bool = False):
        super().__init__()
        
        # We use smp.Segformer which implements the MiT backbone and MLP decoder.
        # mit_b2 is the recommended 'Quality' choice for RTX 50-series.
        self.model = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            classes=num_classes,
        )

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self._full_model_checkpoint_fallback = False

        if self.use_gradient_checkpointing:
            encoder = getattr(self.model, "encoder", None)
            if encoder is not None and hasattr(encoder, "set_grad_checkpointing"):
                encoder.set_grad_checkpointing(True)
            else:
                self._full_model_checkpoint_fallback = True

    def forward(self, x):
        """
        Forward pass. 
        Note: SegFormer output is typically 1/4 of input resolution.
        SMP handles the upsampling to the original input size internally.
        """
        input_size = x.shape[2:]
        
        # Use checkpointing when enabled to reduce activation memory during training.
        if self.training and self.use_gradient_checkpointing and self._full_model_checkpoint_fallback:
            logits = checkpoint(self.model, x, use_reentrant=False)
        else:
            logits = self.model(x)

        if logits is None:
            raise RuntimeError("SegFormer forward returned None logits")
        
        # Ensure output matches input size exactly (handles odd input dimensions)
        if logits.shape[2:] != input_size:
            logits = F.interpolate(
                logits, size=input_size, mode="bilinear", align_corners=False
            )
            
        return logits