"""Loss and metric helpers for segmentation training and evaluation."""

import torch

def dice_loss(pred: torch.Tensor, target: torch.Tensor, num_classes: int, smooth=1e-8):
    pred = torch.softmax(pred, dim=1)

    # VOC uses label 255 as ignore; exclude those pixels from Dice computation.
    valid_mask = (target != 255)
    target = target.clone()
    target[~valid_mask] = 0  # temporary safe value

    # one_hot returns [N, H, W, C], so permute to [N, C, H, W] for channel-wise math.
    target_onehot = torch.nn.functional.one_hot(target, num_classes)
    target_onehot = target_onehot.permute(0, 3, 1, 2).float()

    valid_mask = valid_mask.unsqueeze(1)

    pred = pred * valid_mask
    target_onehot = target_onehot * valid_mask

    intersection = (pred * target_onehot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))

    dice = (2 * intersection + smooth) / (union + smooth)
    # Average only over classes present in the target to avoid skew from absent classes.
    class_present = target_onehot.sum(dim=(2, 3)) > 0
    dice = (dice * class_present).sum(dim=1) / class_present.sum(dim=1).clamp_min(1)
    return 1 - dice.mean()

def compute_means(pred: torch.Tensor, target: torch.Tensor, num_classes: int, smooth = 1e-8):
    target = target.long()
    pred_labels = torch.argmax(pred, dim=1)
    # Ignore VOC's 255 label and compute class stats from a compact confusion matrix.
    valid_mask = (target != 255)
    if not valid_mask.any():
        zero = pred.new_tensor(0.0)
        return zero, zero

    target_valid = target[valid_mask]
    pred_valid = pred_labels[valid_mask]

    indices = target_valid * num_classes + pred_valid
    confusion = torch.bincount(indices, minlength=num_classes * num_classes)
    confusion = confusion.reshape(num_classes, num_classes).float()

    true_positive = confusion.diag()
    pred_sum = confusion.sum(dim=0)
    target_sum = confusion.sum(dim=1)

    dice = (2 * true_positive + smooth) / (pred_sum + target_sum + smooth)
    union = pred_sum + target_sum - true_positive
    iou = (true_positive + smooth) / (union + smooth)

    class_present = (pred_sum + target_sum) > 0
    if not class_present.any():
        zero = pred.new_tensor(0.0)
        return zero, zero

    return dice[class_present].mean(), iou[class_present].mean()