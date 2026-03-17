"""Data Augmentation and preprocessing transforms for Segmentation."""

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class EvalTransforms:
    '''Transforms for evaluation, including resizing and type conversions. 
        Does only resize and type conversions for consistent evaluation.
    '''
    def __init__(self, size=(384, 384)):
        self.size = size

    def __call__(self, image, mask=None):
        if mask is None:
            if isinstance(image, dict):
                sample = image
                image, mask = sample['image'], sample['mask']
                mask = mask - 1
                mask[mask < 0] = 255
            else:
                raise TypeError("Invalid arguments")
        image, mask = tv_tensors.Image(image), tv_tensors.Mask(mask)

        image = F.resize(image, self.size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask = F.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)

        image = F.to_image(image)
        image = F.to_dtype(image, torch.float32, scale=True)
        image = F.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        mask = F.to_image(mask)
        mask = F.to_dtype(mask, torch.int64, scale=False)

        return image, mask

class TrainTransforms:
    '''Data augmentation transforms for training,
    including random resized cropPIng, horizontal flipping, and rotation.
    '''
    def __init__(self, size=(384, 384), scale=(0.5, 1.5), ratio=(1, 1), rotation_degrees=5):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.rotation_degrees = rotation_degrees
        self.flips = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
        ])
        self.random_resized_crop = v2.RandomResizedCrop(size=self.size, scale=self.scale, ratio=self.ratio)
        self.colorjitter = v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05,)

    def __call__(self, image, mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            if isinstance(image, dict):
                sample = image
                image, mask = sample['image'], sample['mask']
            else:
                raise TypeError("Invalid arguments")
        image, mask = tv_tensors.Image(image), tv_tensors.Mask(mask)

        image, mask = self.flips(image, mask)
        image, mask = self.random_resized_crop(image, mask)
        image = self.colorjitter(image)

        #Converting the image to float32 and mask to int64 as only one channel in mask
        image = F.to_image(image)
        image = F.to_dtype(image, torch.float32, scale=True)
        image = F.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        mask = F.to_image(mask)
        mask = F.to_dtype(mask, torch.int64, scale=False)

        return image, mask