"""Data Augmentation and preprocessing transforms for Segmentation."""

import cv2
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2, InterpolationMode
from torchvision.transforms.v2 import functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
    def __init__(self, num_classes, min_area=150, hole_area=150):
        self.num_classes = num_classes
        self.min_area = min_area
        self.hole_area = hole_area

    def __call__(self, logits):
        probs = torch.softmax(logits, dim=1)  # (B, C, H, W)

        if probs.ndim == 4:
            return torch.stack([self._process_single(p) for p in probs])

        return self._process_single(probs)

    def _process_single(self, probs):
        processors = {
            1: PostProcessingRoads(),
            2: PostProcessingBuildings(),
            3: PostProcessingWater(),
        }

        probs = probs.cpu().numpy()  # (C, H, W)

        # 1. Use argmax to lock in the predictions and resolve overlaps
        base_mask = np.argmax(probs, axis=0).astype(np.uint8)
        final_mask = np.zeros_like(base_mask)

        for cls in range(1, self.num_classes):  # Skipping Background class 0
            cls_mask = (base_mask == cls).astype(np.uint8)

            # 2. Extract BOTH external boundaries (islands) and internal holes using CCOMP hierarchy
            contours, hierarchy = cv2.findContours(cls_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            
            if hierarchy is not None:
                hierarchy = hierarchy[0]
                for i in range(len(contours)):
                    is_hole = hierarchy[i][3] != -1
                    area = cv2.contourArea(contours[i])

                    if is_hole and area < self.hole_area:
                        cv2.drawContours(cls_mask, [contours[i]], -1, 1, -1)  # Fill tiny internal hole
                    elif not is_hole and area < self.min_area:
                        cv2.drawContours(cls_mask, [contours[i]], -1, 0, -1)  # Erase tiny external island

            # 3. Use larger kernels for 512x512 images (3x3 is too small)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            cls_mask = cv2.morphologyEx(cls_mask, cv2.MORPH_OPEN, kernel)
            cls_mask = cv2.morphologyEx(cls_mask, cv2.MORPH_CLOSE, kernel)

            cls_mask = processors[cls](torch.from_numpy(cls_mask)).numpy().astype(np.uint8)

            final_mask[cls_mask == 1] = cls

        return torch.from_numpy(final_mask).long()

    def _straighten_edges(self, binary_mask):
        """Takes wavy predictions and snaps them to straight-edged polygons."""
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        straight_mask = np.zeros_like(binary_mask)
        
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            # Epsilon dictates how aggressive the straightening is (0.01 to 0.04 is typical)
            epsilon = 0.02 * cv2.arcLength(c, True) 
            approx = cv2.approxPolyDP(c, epsilon, True)
            cv2.drawContours(straight_mask, [approx], -1, 1, -1)
        
        return straight_mask

class PostProcessingRoads:
    def __init__(self, min_road_area=400, gap_threshold=15, road_width=5):
        """
        Args:
            min_road_area: Removes any isolated road segment smaller than this (pixels).
            gap_threshold: Bridges gaps of this size between road segments (pixels).
            road_width: The fixed width to dilate the roads back to.
        """
        self.min_road_area = min_road_area
        self.gap_threshold = gap_threshold
        self.road_width = road_width

    def __call__(self, road_logits):
        """
        Assumes road_logits is (B, H, W) binary mask or probability map.
        If it's multi-class, slice it for your ROAD index first!
        """
        if road_logits.ndim == 3:
            return torch.stack([self._process_single(r) for r in road_logits])
        return self._process_single(road_logits)

    def _process_single(self, road_prob):
        # Convert torch tensor probability map to binary 
        road_prob = road_prob.cpu().numpy()
        binary_road = (road_prob > 0.5).astype(np.uint8)

        # STAGE 1: Remove Isolated Blobs
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_road)
        clean_mask = np.zeros_like(binary_road)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_road_area:
                clean_mask[labels == i] = 1

        # STAGE 2: Connectivity Enforcement (Bridge Gaps)
        # Using a closing kernel merges road segments that are close together!
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (self.gap_threshold, self.gap_threshold))
        connected_road = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)

        # STAGE 3: Skeletonization
        skeleton = np.zeros_like(connected_road)
        temp = connected_road.copy()
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        # Iterative erosion to strip roads down to 1px spines
        while True:
            eroded = cv2.erode(temp, element)
            temp_open = cv2.dilate(eroded, element)
            temp_open = cv2.subtract(temp, temp_open)
            skeleton = cv2.bitwise_or(skeleton, temp_open)
            temp = eroded.copy()
            if cv2.countNonZero(temp) == 0:
                break

        # STAGE 4: Uniform Dilation
        road_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.road_width, self.road_width))
        final_road = cv2.dilate(skeleton, road_kernel)

        return torch.from_numpy(final_road).long()
    
class PostProcessingBuildings:
    def __init__(self, min_area=100, epsilon_factor=0.02, strict_rectangles=False):
        """
        Args:
            min_area: Removes isolated building specks smaller than this (pixels).
            epsilon_factor: Precision of edge snapping (higher = straighter lines).
            strict_rectangles: If True, forces the mask into a perfect 4-sided bounding box.
        """
        self.min_area = min_area
        self.epsilon_factor = epsilon_factor
        self.strict_rectangles = strict_rectangles
        self.kernel = np.ones((5, 5), np.uint8)  # Larger 5x5 kernel for cleanup

    def __call__(self, building_logits):
        if building_logits.ndim == 3:
            return torch.stack([self._process_single(b) for b in building_logits])
        return self._process_single(building_logits)

    def _process_single(self, building_prob):
        building_prob = building_prob.cpu().numpy()
        binary_build = (building_prob > 0.5).astype(np.uint8)

        # STAGE 1: Morphological Cleanup (Close small holes inside buildings)
        cleaned_mask = cv2.morphologyEx(binary_build, cv2.MORPH_CLOSE, self.kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, self.kernel)

        # STAGE 2: Remove Tiny Connected Components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask)
        area_mask = np.zeros_like(cleaned_mask)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area:
                area_mask[labels == i] = 1

        # STAGE 3: Polygon Regularization (Snap to straight edges)
        contours, _ = cv2.findContours(area_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(area_mask)

        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue

            if self.strict_rectangles:
                # Forces 4-sided bounding boxes (excellent for simple suburbs)
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                cv2.drawContours(final_mask, [box.reshape(-1, 1, 2)], -1, 1, -1)
            else:
                # 📐 Keeps architectural complexity (L-shapes, U-shapes) but straightens wavy edges
                epsilon = self.epsilon_factor * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, epsilon, True)
                cv2.drawContours(final_mask, [approx], -1, 1, -1)

        return torch.from_numpy(final_mask).long()

class PostProcessingWater:
    def __init__(self, min_area=1000, kernel_size=7, blur_sigma=3.0):
        """
        Args:
            min_area: Removes isolated puddles smaller than this (pixels).
            kernel_size: 5 or 7 for closing gaps inside the river network.
            blur_sigma: Gaussian standard deviation. Controls how smooth the coastline is.
        """
        self.min_area = min_area
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.blur_sigma = blur_sigma

    def __call__(self, water_logits):
        if water_logits.ndim == 3:
            return torch.stack([self._process_single(w) for w in water_logits])
        return self._process_single(water_logits)

    def _process_single(self, water_prob):
        water_prob = water_prob.cpu().numpy()
        binary_water = (water_prob > 0.5).astype(np.uint8)

        # STAGE 1: Morphological Closing (Bridge sandbars and fill internal holes)
        closed_mask = cv2.morphologyEx(binary_water, cv2.MORPH_CLOSE, self.kernel)

        # STAGE 2: Area Filtering (Wipe out tiny spurious puddles)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask)
        clean_mask = np.zeros_like(closed_mask)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area:
                clean_mask[labels == i] = 1

        # STAGE 3: Gaussian Melt for Organic Outlines
        if self.blur_sigma > 0:
            # Scale kernel size proportionally to the sigma
            k_size = int(6 * self.blur_sigma + 1)
            if k_size % 2 == 0:
                k_size += 1
            
            # Blur the binary mask to create continuous gradients
            blurred = cv2.GaussianBlur(clean_mask.astype(np.float32), (k_size, k_size), self.blur_sigma)
            
            # Re-threshold at 0.5 to harden the smooth edge back into a crisp binary mask
            final_mask = (blurred > 0.5).astype(np.uint8)
        else:
            final_mask = clean_mask

        return torch.from_numpy(final_mask).long()