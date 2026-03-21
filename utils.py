import logging
from rich.logging import RichHandler
import torch
from torch import nn

# freeze_encoder disables encoder blocks
# get_adamw_param_groups prepares parameter groups for AdamW with differential learning rates and weight decay

logger = logging.getLogger(__name__)

def freeze_encoder(model):
    """
    Prepares an encoder-decoder model for transfer learning.
    Takes exactly one parameter: the model itself.
    """
    
    encoder = getattr(model, "encoder", getattr(getattr(model, "_orig_mod", None), "encoder", None))

    if encoder is None:
        logger.warning("Warning: No 'encoder' attribute found in the model.")
        return

    # 2. Freeze the entire encoder and set to eval mode (locks BatchNorm stats)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    # 3. Target and unfreeze the last two blocks of the encoder
    if hasattr(encoder, "features"):
        trainable_stage_ids = sorted(
            {
                int(name.split(".")[1])
                for name, _ in encoder.named_parameters()
                if name.startswith("features.") and len(name.split(".")) > 2 and name.split(".")[1].isdigit()
            }
        )[-2:]

        # Unfreeze parameters in the last two blocks
        for name, p in encoder.named_parameters():
            if any(name.startswith(f"features.{idx}.") for idx in trainable_stage_ids):
                p.requires_grad = True
                
        # Set ONLY those specific encoder blocks back to train mode
        for idx in trainable_stage_ids:
            block = getattr(encoder.features, str(idx), None)
            if block is not None:
                block.train()

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

def get_adamw_param_groups(model: nn.Module, decoder_lr: float, backbone_lr: float, weight_decay: float) -> list[dict]:
    '''Prepares parameter groups for AdamW optimizer with differential learning rates and weight decay.'''
    dec_decay_params = [] # For weights in the decoder that should have weight decay
    dec_no_weight_decay_params = [] # For biases and normalization parameters in the decoder that should not have weight decay
    enc_decay_params = [] # For weights in the encoder that should have weight decay (if unfrozen)
    enc_no_weight_decay_params = [] # For biases and normalization parameters in the encoder that should not have weight decay (if unfrozen)

    for name, param in model.named_parameters():
        # Stop all but two blocks of encoder
        if not param.requires_grad:
            continue

        # Handles both "encoder.*" and "_orig_mod.encoder.*"
        is_encoder_param = name.startswith("encoder.") or name.startswith("_orig_mod.encoder.")

        # Catches normalization weights and biases
        if param.ndim <= 1 or name.endswith(".bias"):
            if is_encoder_param:
                enc_no_weight_decay_params.append(param)
            else:
                dec_no_weight_decay_params.append(param)
        # Catches all the remaining weights
        else:                                          
            if is_encoder_param:                    
                enc_decay_params.append(param)
            else:
                dec_decay_params.append(param)

    param_groups = []
    if dec_decay_params:
        param_groups.append({"params": dec_decay_params,
                             "weight_decay": weight_decay,
                             "lr": decoder_lr
                             })
    if dec_no_weight_decay_params:
        param_groups.append({"params": dec_no_weight_decay_params,
                             "weight_decay": 0.0,
                             "lr": decoder_lr
                            })
    if enc_decay_params:
        param_groups.append({"params": enc_decay_params,
                             "weight_decay": weight_decay,
                             "lr": backbone_lr
                             })
    if enc_no_weight_decay_params:
        param_groups.append({"params": enc_no_weight_decay_params,
                             "weight_decay": 0.0,
                             "lr": backbone_lr
                             })

    return param_groups