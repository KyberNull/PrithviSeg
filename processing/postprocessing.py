"""Class-aware segmentation postprocessing utilities."""

import cv2
import numpy as np
from skimage.morphology import skeletonize
import torch


class PostProcessing:
    def __init__(self, num_classes, road_thickness=3):
        self.num_classes = num_classes
        self.road_thickness = road_thickness

        # Global class ids: 0 background, 1 roads, 2 buildings, 3 water.
        self.processors = {
            1: PostProcessingRoads(thickness=road_thickness),
            2: PostProcessingBuildings(),
            3: PostProcessingWater(),
        }

    def __call__(self, logits):
        probs = torch.softmax(logits, dim=1)
        if probs.ndim == 4:
            return torch.stack([self._process_single(p) for p in probs])
        return self._process_single(probs)

    def _process_single(self, probs):
        #Creates a mask with only 0s to edit 
        base_mask = torch.argmax(probs, dim=0).cpu().numpy().astype(np.uint8)
        final_mask = np.zeros_like(base_mask)

        #For each class choose a specific function of post-processing, merge the masks and return the finalized_mask
        for cls_id, processor in self.processors.items():
            cls_mask = (base_mask == cls_id).astype(np.uint8)
            if np.any(cls_mask):
                processed_cls = processor(cls_mask)
                if isinstance(processed_cls, torch.Tensor):
                    processed_cls = processed_cls.cpu().numpy()
                final_mask[processed_cls > 0] = cls_id
        return torch.from_numpy(final_mask).long()


class PostProcessingRoads:
    def __init__(self, min_area=200, connect_dist=80, min_branch=2, thickness=5):
        self.min_area = min_area
        self.connect_dist = connect_dist
        self.min_branch = min_branch
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.blur_kernel = (3,3)
        self.thickness = max(1, int(thickness))

    def __call__(self, mask):
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy().astype(np.uint8)
        if mask.sum() == 0:
            return mask

        # Initial denoise and edge smooth.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        soft_mask = cv2.GaussianBlur((mask * 255).astype(np.uint8), self.blur_kernel, 0)
        _, mask = cv2.threshold(soft_mask, 120, 1, cv2.THRESH_BINARY)


        # Remove tiny disconnected road islands before skeletonization.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned = np.zeros_like(mask, dtype=np.uint8)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= self.min_area:
                cleaned[labels == label] = 1
        if cleaned.sum() == 0:
            return mask

        # Keep only meaningful skeleton branches, then regrow to smoother roads.
        skel = skeletonize(cleaned > 0).astype(np.uint8)
        if skel.sum() == 0:
            return cleaned

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skel, connectivity=8)
        skel_clean = np.zeros_like(skel, dtype=np.uint8)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= self.min_branch:
                skel_clean[labels == label] = 1
        if skel_clean.sum() == 0:
            skel_clean = skel

        # Iterative dilation gives a clearer, predictable thickness control.
        grow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        regrown = cv2.dilate(skel_clean, grow_kernel, iterations=self.thickness)
        # Never let postprocessing make roads thinner than the cleaned input mask.
        regrown = np.maximum(regrown, cleaned).astype(np.uint8)

        # Final smooth and binarize for cleaner boundaries.
        regrown = cv2.morphologyEx(regrown, cv2.MORPH_CLOSE, self.kernel)
        regrown_soft = cv2.GaussianBlur((regrown * 255).astype(np.uint8), self.blur_kernel, 0)
        _, regrown = cv2.threshold(regrown_soft, 96, 1, cv2.THRESH_BINARY)
        return (regrown > 0).astype(np.uint8)


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
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(mask)

        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            if self.strict_rectangles:
                rect = cv2.minAreaRect(c)
                pts = np.int32(cv2.boxPoints(rect))
            else:
                epsilon = self.epsilon_factor * cv2.arcLength(c, True)
                pts = cv2.approxPolyDP(c, epsilon, True).astype(np.int32)
            cv2.fillPoly(final_mask, [np.asarray(pts, dtype=np.int32)], 1)

        return final_mask


class PostProcessingWater:
    def __init__(self, min_area=500, kernel_size=7):
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

        final_mask = cv2.medianBlur(final_mask, 7)
        return (final_mask > 0).astype(np.uint8)
