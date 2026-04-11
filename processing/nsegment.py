import cv2
import numpy as np


class NoisySegmentPlus:
    def __init__(self, alpha_sigma_list=None, prob=0.5, area_thresh=1000, ignore_label=255):
        if not alpha_sigma_list:
            alpha_sigma_list = [
                (1, 3),
                (1, 5),
                (1, 10),
                (15, 3),
                (15, 5),
                (15, 10),
                (30, 3),
                (30, 5),
                (30, 10),
                (50, 3),
                (50, 5),
                (50, 10),
                (100, 3),
                (100, 5),
                (100, 10),
            ]

        self.alpha_sigma_list = alpha_sigma_list
        self.prob = prob
        self.area_thresh = area_thresh
        self.ignore_label = ignore_label

    def __call__(self, segment):
        if np.random.rand() > self.prob:
            return segment

        return self.transform(segment)

    def transform(self, segment, random_state=None):
        if random_state is None:
            random_state = np.random.RandomState(None)

        segment_2d = np.asarray(segment)
        if segment_2d.ndim == 3:
            if segment_2d.shape[0] == 1:
                segment_2d = segment_2d[0]
            elif segment_2d.shape[-1] == 1:
                segment_2d = segment_2d[..., 0]
            else:
                # Semantic masks should be single-channel; fall back to first channel.
                segment_2d = segment_2d[..., 0]
        elif segment_2d.ndim != 2:
            segment_2d = np.squeeze(segment_2d)
            if segment_2d.ndim != 2:
                raise ValueError(f"Expected 2D mask, got shape {np.asarray(segment).shape}")

        # Generate random displacement fields
        shape = segment_2d.shape
        dx = 2 * random_state.rand(*shape) - 1
        dy = 2 * random_state.rand(*shape) - 1

        # Apply stochastic Gaussian smoothing
        choice = random_state.randint(0, len(self.alpha_sigma_list))
        alpha, sigma = self.alpha_sigma_list[choice]
        dx, dy = alpha * dx, alpha * dy

        # Scale-aware deformation suppression
        mask_ignore = np.zeros(shape, dtype=bool)
        unique_labels = np.unique(segment_2d)

        for class_id in unique_labels:
            if class_id == -1 or class_id == self.ignore_label:
                continue

            class_mask = np.ascontiguousarray((segment_2d == class_id).astype(np.uint8))
            num_labels, labels = cv2.connectedComponents(class_mask)

            for comp_id in range(1, num_labels):
                comp_mask = (labels == comp_id).astype(np.uint8)
                area = np.sum(comp_mask)
                if area > self.area_thresh:
                    continue

                ys, xs = np.where(comp_mask)
                y1, x1 = ys.min(), xs.min()
                y2, x2 = ys.max(), xs.max()

                pad = alpha // 2
                y1_pad = max(y1 - pad, 0)
                y2_pad = min(y2 + pad, shape[0] - 1)
                x1_pad = max(x1 - pad, 0)
                x2_pad = min(x2 + pad, shape[1] - 1)
                mask_ignore[y1_pad : y2_pad + 1, x1_pad : x2_pad + 1] = True

        dx[mask_ignore], dy[mask_ignore] = 0, 0
        dx = cv2.GaussianBlur(dx, (0, 0), sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), sigma)

        # Label-specific deformation
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x = np.clip(x + dx, 0, shape[1] - 1).astype(np.float32)
        map_y = np.clip(y + dy, 0, shape[0] - 1).astype(np.float32)

        noisy_segment = cv2.remap(
            segment_2d,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
        )
        return noisy_segment