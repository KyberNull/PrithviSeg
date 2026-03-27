"""Loss and metric helpers for segmentation training and evaluation."""

import torch
import segmentation_models_pytorch as smp

focal_loss = smp.losses.FocalLoss(
    mode="multiclass",   # important for your case
    gamma=2.0,
)

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

def iou_metric(pred: torch.Tensor, target: torch.Tensor, num_classes: int, smooth = 1e-8):
    target = target.long()
    pred_labels = torch.argmax(pred, dim=1)
    # Ignore VOC's 255 label and compute class stats from a compact confusion matrix.
    valid_mask = (target != 255)
    if not valid_mask.any():
        return pred.new_tensor(0.0)

    target_valid = target[valid_mask]
    pred_valid = pred_labels[valid_mask]

    indices = target_valid * num_classes + pred_valid
    confusion = torch.bincount(indices, minlength=num_classes * num_classes)
    confusion = confusion.reshape(num_classes, num_classes).float()

    true_positive = confusion.diag()
    pred_sum = confusion.sum(dim=0)
    target_sum = confusion.sum(dim=1)

    union = pred_sum + target_sum - true_positive
    iou_per_class = (true_positive + smooth) / (union + smooth)

    class_present = (pred_sum + target_sum) > 0
    present_count = class_present.sum().clamp_min(1)

    iou = (iou_per_class * class_present).sum() / present_count
    return iou

def iou_metric_processed_fast(pred, target, num_classes, eps=1e-6):
    """
    pred: (B, H, W) - processed predictions
    target: (B, H, W)
    """

    device = pred.device
    target = target.to(device)

    pred = pred.view(-1)
    target = target.view(-1)

    # Clamp just in case
    pred = pred.long()
    target = target.long()

    # Build confusion matrix
    cm = torch.bincount(
        num_classes * target + pred,
        minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)

    # Intersection = diagonal
    intersection = cm.diag()

    # Union = sum over row + col - intersection
    union = cm.sum(1) + cm.sum(0) - intersection

    iou = (intersection + eps) / (union + eps)

    return iou.mean()