"""Data Augmentation and preprocessing transforms for VOC segmentation."""

import random
import torch
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class VOCEvalTransforms:
    '''Transforms for evaluation, including resizing and type conversions.'''
    #Does only resize and type conversions for consistent evaluation
    def __init__(self, size=(256, 256)):
        self.size = size

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        image = F.resize(image, self.size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask = F.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)

        image = F.to_image(image)
        image = F.to_dtype(image, torch.float32, scale=True)
        image = F.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        mask = F.to_image(mask)
        mask = F.to_dtype(mask, torch.int64, scale=False)

        return image, mask

class VOCTrainTransforms:
    '''Data augmentation transforms for training, including random resized cropping, horizontal flipping, and rotation.'''
    def __init__(self, size=(256, 256), scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3), rotation_degrees=5):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.rotation_degrees = rotation_degrees

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        #Gets the paramters and resizes and interpolates the images and masks
        i, j, h, w = v2.RandomResizedCrop.get_params(image, self.scale, self.ratio)
        image = F.resized_crop(
            image,
            top=i,
            left=j,
            height=h,
            width=w,
            size=self.size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = F.resized_crop(
            mask,
            top=i,
            left=j,
            height=h,
            width=w,
            size=self.size,
            interpolation=InterpolationMode.NEAREST,
        )

        if random.random() < 0.5:
            image = F.horizontal_flip(image)
            mask = F.horizontal_flip(mask)

        #Implementing the rotation for both image and mask, mask the fill is 255 by the dataset VOCSegmentation and image it is 0 (background)
        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        image = F.rotate(image, angle=angle, interpolation=InterpolationMode.BILINEAR, fill=0) #type: ignore
        mask = F.rotate(mask, angle=angle, interpolation=InterpolationMode.NEAREST, fill=255) #type: ignore

        #Converting the image to float32 and mask to int64 as only one channel in mask
        image = F.to_image(image)
        image = F.to_dtype(image, torch.float32, scale=True)
        image = F.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        mask = F.to_image(mask)
        mask = F.to_dtype(mask, torch.int64, scale=False)

        return image, mask
