import logging
from rich.logging import RichHandler
import torch
from torch import nn


# FIXME: Implement it properly
def freeze_encoder(model, encoder_lr):
    """Freeze encoder weights and normalisation weights except the last two blocks for transfer learning."""
    
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        encoder = getattr(getattr(model, "_orig_mod", None), "encoder", None)
    if encoder is None:
        return
    
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    # EfficientNetV2-S encoder blocks are under encoder.features.<idx>; keep the last two trainable.

    for i in range(2):
        block = encoder.features[len(encoder.features) - 1 - i]
        block.eval() 

        for name, m in block.named_modules():
            if not isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                for p in m.parameters():
                    p.requires_grad = True

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler()],
        force=True,
    )

def device_setup():
    device = torch.device('cpu')
    pin_memory = False
    amp_dtype = torch.bfloat16

    if torch.cuda.is_available():
        device = torch.device('cuda')
        pin_memory = True
        amp_dtype = torch.float16
        torch.backends.cudnn.benchmark = True

    elif torch.mps.is_available():
        device = torch.device('mps')
        amp_dtype = torch.float16

    return device, pin_memory, amp_dtype

def save_checkpoint(model, optimizer, scheduler, scaler, epoch: int, path: str):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
    }, path)

def get_adamw_param_groups(model: nn.Module, learning_rate: float, backbone_lr: float, weight_decay: float) -> list[dict]:
    head_decay_params = []
    head_no_decay_params = []
    enc_decay_params = []
    enc_no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:                  #stops all but two blocks of encoder
            continue
        is_encoder_param = name.startswith("encoder.")
        if param.ndim <= 1 or name.endswith(".bias"): #catches are normalization weights and biases
            if is_encoder_param:
                enc_no_decay_params.append(param)
            else:
                head_no_decay_params.append(param)
        else:                                          #catches all the weights 
            if is_encoder_param:                    
                enc_decay_params.append(param)
            else:
                head_decay_params.append(param)

    param_groups = []
    if head_decay_params:
        param_groups.append({"params": head_decay_params, "weight_decay": weight_decay, "lr": learning_rate})
    if head_no_decay_params:
        param_groups.append({"params": head_no_decay_params, "weight_decay": 0.0, "lr": learning_rate})
    if enc_decay_params:
        param_groups.append({"params": enc_decay_params, "weight_decay": weight_decay, "lr": backbone_lr})
    if enc_no_decay_params:
        param_groups.append({"params": enc_no_decay_params, "weight_decay": 0.0, "lr": backbone_lr})

    return param_groups