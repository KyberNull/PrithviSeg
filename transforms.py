"""Data Augmentation and preprocessing transforms for Segmentation."""

import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import convolve, median_filter
from scipy.spatial import cKDTree
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
            1: PostProcessingWater(),
            2: PostProcessingBuildings(),
            3: PostProcessingRoads(),
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
    def __init__(self, min_area=800, connect_dist=100, min_branch=50):
        self.min_area = min_area
        self.connect_dist = connect_dist
        self.min_branch = min_branch
        self.kernel = np.ones((5, 5), np.uint8)

    def __call__(self, mask):
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy().astype(np.uint8)

        if mask.sum() == 0:
            return mask

        # --------------------------------------------------
        # 1. CLEAN MASK (morphology + remove noise)
        # --------------------------------------------------
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = remove_small_objects(mask.astype(bool), min_size=self.min_area)
        mask = mask.astype(np.uint8)

        # --------------------------------------------------
        # 2. DISTANCE MAP (road width estimate)
        # --------------------------------------------------
        dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        dist_map = median_filter(dist_map, size=5)

        # --------------------------------------------------
        # 3. SKELETON (centerline)
        # --------------------------------------------------
        skeleton = skeletonize(mask > 0)
        skeleton = skeleton.astype(np.uint8)

        # --------------------------------------------------
        # 4. REMOVE SMALL BRANCHES (graph pruning)
        # --------------------------------------------------
        skeleton = remove_small_objects(skeleton.astype(bool), min_size=self.min_branch)
        skeleton = skeleton.astype(np.uint8)

        # --------------------------------------------------
        # 5. CONNECT BROKEN ROADS (topology repair)
        # --------------------------------------------------
        skeleton = self._connect_endpoints(skeleton)

        # --------------------------------------------------
        # 6. RECONSTRUCT ROAD WIDTH (buffering)
        # --------------------------------------------------
        num_labels, labels = cv2.connectedComponents(skeleton)

        reconstructed = np.zeros_like(mask, dtype=np.uint8)

        for label in range(1, num_labels):
            component = (labels == label)

            ys, xs = np.where(component)
            if len(ys) < 20:
                continue

            widths = dist_map[ys, xs]
            widths = widths[widths > 0]

            if len(widths) == 0:
                continue

            radius = int(np.median(widths))

            if radius < 1:
                continue

            for y, x in zip(ys, xs):
                cv2.circle(reconstructed, (x, y), radius, 1, -1)

        # --------------------------------------------------
        # 7. FINAL CLEANUP (OUTSIDE LOOP)
        # --------------------------------------------------
        reconstructed = cv2.morphologyEx(reconstructed, cv2.MORPH_CLOSE, self.kernel)

        return reconstructed.astype(np.uint8)

    # ======================================================
    # 🔥 ENDPOINT CONNECTION (Topology Repair)
    # ======================================================
    def _connect_endpoints(self, skel):
        endpoints = self._find_endpoints(skel)

        if len(endpoints) < 2:
            return skel

        tree = cKDTree(endpoints)

        for i, p in enumerate(endpoints):
            dists, idxs = tree.query(p, k=4, distance_upper_bound=self.connect_dist)

            for j in idxs:
                if j == i or j >= len(endpoints):
                    continue

                q = endpoints[j]

                # avoid tiny loops
                if np.linalg.norm(p - q) < 10:
                    continue

                cv2.line(
                    skel,
                    (int(p[1]), int(p[0])),
                    (int(q[1]), int(q[0])),
                    1,
                    1
                )

        return skel

    # ======================================================
    # 🔥 ENDPOINT DETECTION
    # ======================================================
    def _find_endpoints(self, skel):
        kernel = np.array([
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1]
        ])

        conv = convolve(skel, kernel, mode='constant', cval=0)
        endpoints = (conv == 11)

        return np.column_stack(np.where(endpoints))
    
class PostProcessingBuildings:
    def __init__(self, min_area=100, epsilon_factor=0.015, strict_rectangles=False):
        self.min_area = min_area
        self.epsilon_factor = epsilon_factor
        self.strict_rectangles = strict_rectangles
        self.kernel = np.ones((5, 5), np.uint8)

    def __call__(self, mask):
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy().astype(np.uint8)

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
    def __init__(self, min_area=1000, kernel_size=7):
        self.min_area = min_area
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def __call__(self, mask):
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy().astype(np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(mask)

        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            cv2.fillPoly(final_mask, [c], 1)

        # smoother than gaussian thresholding
        final_mask = cv2.medianBlur(final_mask, 7)

        return (final_mask > 0).astype(np.uint8)