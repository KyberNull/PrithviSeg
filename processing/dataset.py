"""Custom geospatial dataset class."""

import logging

import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors


class GeospatialDataset(Dataset):
    """Custom dataset class to load the geospatial dataset."""

    def __init__(self, img_dir, img_mask, transform=None):
        self.img_dir = img_dir
        self.img_mask = img_mask
        self.transform = transform
        self.logger = logging.getLogger(__name__)

        self.images = sorted(
            filename
            for filename in os.listdir(img_dir)
            if os.path.isfile(os.path.join(img_dir, filename))
        )
        self.masks = sorted(
            filename
            for filename in os.listdir(img_mask)
            if os.path.isfile(os.path.join(img_mask, filename))
        )

        image_by_stem = {os.path.splitext(name)[0]: name for name in self.images}
        mask_by_stem = {os.path.splitext(name)[0]: name for name in self.masks}
        common_stems = sorted(set(image_by_stem).intersection(mask_by_stem))
        self.pairs = [(image_by_stem[stem], mask_by_stem[stem]) for stem in common_stems]

        missing_masks = sorted(set(image_by_stem) - set(mask_by_stem))
        missing_images = sorted(set(mask_by_stem) - set(image_by_stem))

        if missing_masks or missing_images:
            self.logger.warning(
                "Dataset pair mismatch in %s and %s: %d images, %d masks, %d usable pairs",
                img_dir,
                img_mask,
                len(self.images),
                len(self.masks),
                len(self.pairs),
            )
            if missing_masks:
                self.logger.warning(
                    "Ignoring %d images without masks (sample: %s)",
                    len(missing_masks),
                    ", ".join(missing_masks[:5]),
                )
            if missing_images:
                self.logger.warning(
                    "Ignoring %d masks without images (sample: %s)",
                    len(missing_images),
                    ", ".join(missing_images[:5]),
                )

        if not self.pairs:
            raise ValueError(
                f"No matched image/mask pairs found in {img_dir} and {img_mask}"
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_name, mask_name = self.pairs[idx]
        img_path = os.path.join(self.img_dir, image_name)
        mask_path = os.path.join(self.img_mask, mask_name)
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        image, mask = tv_tensors.Image(image), tv_tensors.Mask(mask)

        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(np.array(mask)).long()

        # Keep mask layout consistent across samples so default_collate can stack safely.
        mask = torch.as_tensor(mask)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 3:
            if mask.shape[0] == 1:
                pass
            elif mask.shape[-1] == 1:
                mask = mask.permute(2, 0, 1)
            else:
                mask = mask[:1, ...]
        else:
            mask = mask.squeeze()
            if mask.ndim != 2:
                raise ValueError(f"Expected mask with 2 or 3 dims, got shape {tuple(mask.shape)}")
            mask = mask.unsqueeze(0)

        mask = mask.float()
        return image, mask
