"""Data Augmentation and preprocessing transforms for Segmentation."""

import cv2
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2, InterpolationMode
from torchvision.transforms.v2 import functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

#TODO: Use GPU Accelerated CLAHE with OpenCV's CUDA module for faster preprocessing.
def apply_clahe(image):
    # image: torch tensor [C, H, W] in float32 [0,1]
    img = image.permute(1, 2, 0).cpu().numpy()  # HWC

    img = (img * 255).astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)

class EvalTransforms:
    '''Transforms for evaluation, including resizing and type conversions. 
        Does only resize and type conversions for consistent evaluation.
    '''
    def __init__(self, size=(512, 512)):
        self.size = size

    def __call__(self, image, mask=None):
        if mask is None:
            if isinstance(image, dict):
                sample = image
                image, mask = sample['image'], sample['mask']
            else:
                raise TypeError("Invalid arguments")
        image, mask = tv_tensors.Image(image), tv_tensors.Mask(mask)
        image = F.to_dtype(image, torch.float32, scale=True)

        image = torch.nn.functional.interpolate(image.unsqueeze(0), self.size, mode="area", antialias=False).squeeze(0)
        mask = F.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)

        image = F.to_image(image)
        image = apply_clahe(image)
        image = F.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        mask = mask.to(torch.int64)

        return image, mask

class TrainTransforms:
    '''Data augmentation transforms for training,
    including random resized cropPIng, horizontal flipping, and rotation.
    '''
    def __init__(self, size=(512, 512), rotation_degrees=10):
        self.size = size
        self.rotation_degrees = rotation_degrees
        self.flips = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
        ])
        self.rotate90 = v2.RandomChoice([
            v2.RandomRotation((0, 0)),
            v2.RandomRotation((90, 90)),
            v2.RandomRotation((180, 180)),
            v2.RandomRotation((270, 270)),
        ])
      
    def __call__(self, image, mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            if isinstance(image, dict):
                sample = image
                image, mask = sample['image'], sample['mask']
            else:
                raise TypeError("Invalid arguments")
        image, mask = tv_tensors.Image(image), tv_tensors.Mask(mask)
        image = F.to_dtype(image, torch.float32, scale=True)

        image = torch.nn.functional.interpolate(image.unsqueeze(0), self.size, mode="area", antialias=False).squeeze(0)
        mask = F.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)
        image, mask = self.flips(image, mask)
        image, mask = self.rotate90(image, mask)
        
        #Converting the image to float32 and mask to int64 as only one channel in mask
        image = F.to_image(image)
        image = apply_clahe(image)
        image = F.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        mask = mask.to(torch.int64)

        return image, mask

class PostProcessing:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        # Map class IDs to their specialized processors
        self.processors = {
            1: PostProcessingRoads(),
            2: PostProcessingBuildings(),
            3: PostProcessingWater(),
        }

    def __call__(self, logits):
        probs = torch.softmax(logits, dim=1)
        if probs.ndim == 4:
            return torch.stack([self._process_single(p) for p in probs])
        return self._process_single(probs)

    def _process_single(self, probs):
        # 1. Argmax to get the base layout (Numpy)
        base_mask = torch.argmax(probs, dim=0).cpu().numpy().astype(np.uint8)
        final_mask = np.zeros_like(base_mask)

        # 2. Process each class individually
        for cls_id, processor in self.processors.items():
            # Extract mask for current class
            cls_mask = (base_mask == cls_id).astype(np.uint8)
            
            if np.any(cls_mask):
                # Run the specific processor (Roads/Buildings/Water)
                # Note: Keeping them as numpy inside for speed
                processed_cls = processor(cls_mask)
                
                # Convert back to numpy if processor returns tensor
                if isinstance(processed_cls, torch.Tensor):
                    processed_cls = processed_cls.cpu().numpy()
                
                # Burn the processed result into the final map
                final_mask[processed_cls > 0] = cls_id

        return torch.from_numpy(final_mask).long()

    def _safe_draw(self, mask, contour, color=1):
        """Helper to prevent the 'No overloads' error."""
        # Ensure contour is int32 and wrapped in a list
        pts = [contour.astype(np.int32)]
        cv2.drawContours(mask, pts, -1, color, thickness=-1)

class PostProcessingRoads:
    def __init__(self, min_area=500, close_kernel_size=7, smooth_sigma=2):
        self.min_area = min_area
        self.kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
        self.smooth_sigma = smooth_sigma

    def __call__(self, mask):
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy().astype(np.uint8)
            
        # 1. Close gaps first
        refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        
        # 2. Gaussian Smoothing - this "melts" the jagged edges into smooth curves
        # ksize must be odd
        ksize = int(6 * self.smooth_sigma + 1)
        if ksize % 2 == 0: ksize += 1
        
        # We apply the blur to the mask (0s and 1s)
        # This creates a "gradient" edge
        smoothed = cv2.GaussianBlur(refined_mask.astype(np.float32), (ksize, ksize), self.smooth_sigma)
        
        # 3. Re-threshold to get hard edges back
        # Higher threshold (e.g., 0.5) makes roads thinner; lower makes them thicker.
        _, smooth_mask = cv2.threshold(smoothed, 0.4, 1, cv2.THRESH_BINARY)
        smooth_mask = smooth_mask.astype(np.uint8)

        # 4. Final Clean-up (Area check)
        contours, _ = cv2.findContours(smooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(mask)

        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            
            # Use a very small epsilon (0.001) just to reduce point count 
            # without changing the smooth shape
            epsilon = 0.001 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            cv2.fillPoly(final_mask, [approx], 1)

        return final_mask
    
class PostProcessingBuildings:
    def __init__(self, min_area=100, epsilon_factor=0.015, strict_rectangles=False):
        self.min_area = min_area
        self.epsilon_factor = epsilon_factor
        self.strict_rectangles = strict_rectangles
        self.kernel = np.ones((5, 5), np.uint8)

    def __call__(self, mask):
        # Handle Tensor vs Numpy
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy().astype(np.uint8)

        # 1. Morphological Cleanup (fills small holes/gaps)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        
        # 2. Extract and Filter contours by area
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(mask)

        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue

            if self.strict_rectangles:
                # Forced 4-point bounding box
                rect = cv2.minAreaRect(c)
                pts = np.int32(cv2.boxPoints(rect))
            else:
                # Architectural simplification (L-shapes, etc.)
                epsilon = self.epsilon_factor * cv2.arcLength(c, True)
                pts = cv2.approxPolyDP(c, epsilon, True).astype(np.int32)

            # 3. Robust drawing
            cv2.fillPoly(final_mask, [pts], 1)

        return final_mask

class PostProcessingWater:
    def __init__(self, min_area=1000, kernel_size=7, blur_sigma=3.0):
        self.min_area = min_area
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.blur_sigma = blur_sigma

    def __call__(self, mask):
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy().astype(np.uint8)

        # 1. Bridge gaps (sandbars/rivers)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        # 2. Filter tiny puddles and smooth outlines
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(mask)

        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            cv2.fillPoly(final_mask, [c], 1)

        # 3. Organic Smoothing (Gaussian Melt)
        if self.blur_sigma > 0:
            k_size = int(6 * self.blur_sigma + 1) | 1 # Ensure odd kernel size
            final_mask = cv2.GaussianBlur(final_mask.astype(np.float32), (k_size, k_size), self.blur_sigma)
            final_mask = (final_mask > 0.5).astype(np.uint8)

        return final_mask