import logging
from rich.logging import RichHandler
import torch

# freeze_encoder disables encoder blocks
# get_adamw_param_groups prepares parameter groups for AdamW with differential learning rates and weight decay

logger = logging.getLogger(__name__)

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
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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

def get_adamw_param_groups(model, weight_decay: float) -> list[dict]:
    '''Prepares parameter groups for AdamW optimizer with differential learning rates and weight decay.'''
    weight_decay_params = [] # For weights in the decoder that should have weight decay
    no_weight_decay_params = [] # For biases and normalization parameters in the decoder that should not have weight decay

    for name, param in model.named_parameters():

        # Catches normalization weights, and biases
        if param.ndim <= 1 or name.endswith(".bias"):
            no_weight_decay_params.append(param)

        # Catches all the remaining weights
        else:
            weight_decay_params.append(param)

    param_groups = []
    if weight_decay_params:
        param_groups.append({"params": weight_decay_params, "weight_decay": weight_decay})
    if no_weight_decay_params:
        param_groups.append({"params": no_weight_decay_params, "weight_decay": 0.0 })

    return param_groups