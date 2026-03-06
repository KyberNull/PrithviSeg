"""Custom geospatial_dataset class"""

from torch.utils.data import Dataset
from transforms import TrainTransforms
import torch
from PIL import Image
from torchvision import tv_tensors
import os

class geospatial_dataset(Dataset):
    """Custom dataset class to load the geospatial dataset"""
    def __init__(self, img_dir, img_mask, transform=None):
        """Declaring all the required attributes"""

        self.img_dir = img_dir
        self.img_mask = img_mask
        self.transform = transform

        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(img_mask))

    def __len__(self):
        """Returns the lenght of the dataset"""
        return len(self.images)

    
    def __getitem__(self, idx):
        """Returns one image and corresponding mask based on index"""
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.img_mask, self.masks[idx])

        image = Image.open(img_path).convert('RGB') 
        mask = Image.open(mask_path).convert('L')


        image = tv_tensors.Image(image)
        mask = tv_tensors.Mask(mask)

        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            import torchvision.transforms.v2.functional as F
            image = F.to_dtype(image, torch.float32, scale=True)
            mask = F.to_dtype(mask, torch.long)

        mask = mask.float()
        if mask.max() > 1:
            mask = mask / 255.0

        return image, mask



