"""Class-aware segmentation postprocessing utilities."""

import cv2
import numpy as np
from scipy.ndimage import convolve, median_filter
from scipy.spatial import KDTree
from skimage.morphology import remove_small_objects, skeletonize
import torch


class PostProcessing:
    def __init__(self, num_classes):
        self.num_classes = num_classes

        # Global class ids: 0 background, 1 roads, 2 buildings, 3 water.
        self.processors = {1: PostProcessingRoads(), 2: PostProcessingBuildings(), 3: PostProcessingWater()}

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

    
    def _safe_draw(self, mask, contour, color=1):
        pts = [contour.astype(np.int32)]
        cv2.drawContours(mask, pts, -1, color, thickness=1)


class PostProcessingRoads:
    def __init__(self, min_area=64, connect_dist=80, min_branch=10):
        self.min_area = min_area
        self.connect_dist = connect_dist
        self.min_branch = min_branch
        self.kernel = np.ones((3, 3), np.uint8)

    def __call__(self, mask):
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy().astype(np.uint8)
        if mask.sum() == 0:
            return mask
        original_mask = mask.copy()

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = remove_small_objects(mask.astype(bool), min_size=self.min_area)
        mask = mask.astype(np.uint8)

        # If pruning removed everything, keep the model prediction.
        if mask.sum() == 0:
            return original_mask

        dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        dist_map = median_filter(dist_map, size=5)
        skeleton = skeletonize(mask > 0)
        skeleton = skeleton.astype(np.uint8)
        skeleton = remove_small_objects(skeleton.astype(bool), min_size=self.min_branch)
        skeleton = skeleton.astype(np.uint8)

        # Skeleton pruning can be too aggressive for thin roads.
        if skeleton.sum() == 0:
            return mask

        skeleton = self._connect_endpoints(skeleton)
        num_labels, labels = cv2.connectedComponents(skeleton)
        reconstructed = np.zeros_like(mask, dtype=np.uint8)

        for label in range(1, num_labels):
            component = labels == label
            ys, xs = np.where(component)
            if len(ys) < 8:
                continue
            widths = dist_map[ys, xs]
            widths = widths[widths > 0]
            if len(widths) == 0:
                continue
            radius = max(1, int(np.round(np.median(widths))))
            if radius < 1:
                continue
            for y, x in zip(ys, xs):
                cv2.circle(reconstructed, (x, y), radius, 1, -1)

        reconstructed = cv2.morphologyEx(reconstructed, cv2.MORPH_CLOSE, self.kernel)

        # Preserve cleaned mask if reconstruction is empty.
        if reconstructed.sum() == 0:
            return mask

        return reconstructed.astype(np.uint8)

    def _connect_endpoints(self, skel):
        endpoints = self._find_endpoints(skel)
        if len(endpoints) < 2:
            return skel
        tree = KDTree(endpoints)

        for i, p in enumerate(endpoints):
            dists, idxs = tree.query(p, k=4, distance_upper_bound=self.connect_dist)
            for j in np.atleast_1d(idxs):
                if j == i or j >= len(endpoints):
                    continue
                q = endpoints[j]
                if np.linalg.norm(p - q) < 10:
                    continue
                cv2.line(skel, (int(p[1]), int(p[0])), (int(q[1]), int(q[0])), 1, 1)

        return skel

    def _find_endpoints(self, skel):
        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
        conv = convolve(skel, kernel, mode="constant", cval=0)
        endpoints = conv == 11
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
