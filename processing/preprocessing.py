from processing import IMAGENET_MEAN, IMAGENET_STD
import torch
import kornia

def get_valid_mask(image: torch.Tensor) -> torch.Tensor:
    """Identifies real imagery vs. legal black-outs (NoData)."""
    return (image.sum(dim=1, keepdim=True) > 1e-4).float()

def apply_kornia_enhancements(lab: torch.Tensor, mask: torch.Tensor, use_shadow: bool) -> torch.Tensor:
    """Core enhancement logic: CLAHE + Optional Shadow Recovery."""
    l, a, b = torch.chunk(lab, 3, dim=1)

    if use_shadow:
        # Avoid skewing stats with black legal voids
        l_stats = torch.where(mask.bool(), l, torch.tensor(100.0, device=l.device))
        flattened_l = l_stats.view(l.size(0), -1)
        thresh = torch.quantile(flattened_l, 0.35, dim=1, keepdim=True).view(-1, 1, 1, 1)
        
        shadow_mask = ((l < thresh) & mask.bool()).float()
        
        eps = 1e-6
        mean_shadow = (l * shadow_mask).sum(dim=(2,3), keepdim=True) / (shadow_mask.sum(dim=(2,3), keepdim=True) + eps)
        mean_light = (l * (mask - shadow_mask)).sum(dim=(2,3), keepdim=True) / ((mask - shadow_mask).sum(dim=(2,3), keepdim=True) + eps)
        
        scale = torch.clamp(mean_light / (mean_shadow + eps), 1.0, 2.5)
        l = torch.where(shadow_mask.bool(), l * scale, l)

    # Always apply CLAHE - best for drone detail
    l_norm = torch.clamp(l / 100.0, 0.0, 1.0)
    l_enhanced = kornia.enhance.equalize_clahe(l_norm, clip_limit=1.5, grid_size=(8, 8))
    
    # Re-apply mask to L channel to keep voids perfectly black
    l_final = (l_enhanced * 100.0) * mask
    return torch.cat([l_final, a, b], dim=1)

def apply_preprocess(batch_rgb: torch.Tensor) -> torch.Tensor:
    """
    Preprocesses drone imagery:
    1. Detects black/missing regions.
    2. Analyzes imagery to decide if shadow recovery is needed.
    3. Enhances in Lab space.
    4. Returns final RGB batch.
    """
    is_single = batch_rgb.ndim == 3
    if is_single:
        batch_rgb = batch_rgb.unsqueeze(0)
    
    device = batch_rgb.device

    # Identify valid imagery vs black borders
    mask = get_valid_mask(batch_rgb)
    lab = kornia.color.rgb_to_lab(batch_rgb)
    
    # Scoring: Is the image too dark? (Using L channel directly)
    l = lab[:, 0:1]
    valid_count = mask.sum(dim=(2,3), keepdim=True) + 1e-6
    shadow_score = ((l < 30.0) * mask).sum(dim=(2,3), keepdim=True) / valid_count
    
    # Decision: Threshold for shadow correction
    use_shadow = shadow_score.mean().item() > 0.3
    
    # Process
    lab_enhanced = apply_kornia_enhancements(lab, mask, use_shadow)
    
    # Convert back to RGB
    result_rgb = kornia.color.lab_to_rgb(lab_enhanced)
    
    imagenet_mean = torch.tensor(IMAGENET_MEAN).to(device)
    imagenet_std = torch.tensor(IMAGENET_STD).to(device)
    
    normalized_rgb = kornia.enhance.normalize(
            result_rgb, 
            imagenet_mean,
            imagenet_std
        )
    
    return normalized_rgb.squeeze(0) if is_single else normalized_rgb