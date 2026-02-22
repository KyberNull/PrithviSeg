import torch

def dice_loss(pred, target, num_classes, smooth=1e-8):
    pred = torch.softmax(pred, dim=1)

    valid_mask = (target != 255)
    target = target.clone()
    target[~valid_mask] = 0  # temporary safe value

    target_onehot = torch.nn.functional.one_hot(target, num_classes)
    target_onehot = target_onehot.permute(0, 3, 1, 2).float()

    valid_mask = valid_mask.unsqueeze(1)

    pred = pred * valid_mask
    target_onehot = target_onehot * valid_mask

    intersection = (pred * target_onehot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))

    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def compute_means(pred, target, num_classes, smooth = 1e-8):
    target = target.long()

    pred = torch.softmax(pred, dim=1)

    valid_mask = (target != 255)
    safe_target = target.clone()
    safe_target[~valid_mask] = 0

    target_onehot = torch.nn.functional.one_hot(safe_target, num_classes)
    target_onehot = target_onehot.permute(0, 3, 1, 2).float()

    valid_mask = valid_mask.unsqueeze(1)

    pred = pred * valid_mask
    target_onehot = target_onehot * valid_mask

    intersection = (pred * target_onehot).sum(dim=(2, 3))
    pred_sum = pred.sum(dim=(2, 3))
    target_sum = target_onehot.sum(dim=(2, 3))

    dice = (2 * intersection + smooth) / (pred_sum + target_sum + smooth)
    union = pred_sum + target_sum - intersection
    iou = (intersection + smooth) / (union + smooth)

    return dice.mean(), iou.mean()