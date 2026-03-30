import os
import re
import torch
import logging
import rasterio
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from PIL import Image as PILImage
from rasterio.windows import Window
from rasterio import features
from shapely.geometry import shape
from shapely.validation import make_valid
from torchvision import tv_tensors
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Import your specific Model/Transforms ---
from model import SegFormer
from transforms import EvalTransforms, PostProcessing

logging.basicConfig(level=logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PILImage.MAX_IMAGE_PIXELS = None

# --- Configuration ---
PATCH_SIZE = 1024
STRIDE = PATCH_SIZE 
NUM_CLASSES = 4 
MODEL_PATH = "model.pt"
USE_TORCH_COMPILE = True

# Folder Setup
DATASET_DIR = "processed_datasets"
MASK_DIR = "processed_masks"
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)


def _adapt_state_dict_for_model(state_dict, model):
    """Align checkpoint key prefixes with current model keys (compiled vs non-compiled)."""
    model_has_orig = any(k.startswith("_orig_mod.") for k in model.state_dict().keys())
    ckpt_has_orig = any(k.startswith("_orig_mod.") for k in state_dict.keys())

    if ckpt_has_orig and not model_has_orig:
        return {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in state_dict.items()
        }

    if model_has_orig and not ckpt_has_orig:
        return {f"_orig_mod.{k}": v for k, v in state_dict.items()}

    return state_dict

def vectorize_chunk(args):
    """Processes a small chunk of the predicted data into polygons"""
    chunk_mask, chunk_transform, class_val = args
    binary_mask = (chunk_mask == class_val).astype(np.uint8)
    if not np.any(binary_mask): return []
    
    shapes = features.shapes(chunk_mask, mask=binary_mask, transform=chunk_transform)
    geometries = []
    for geom, _ in shapes:
        try:
            s = shape(geom)
            if not s.is_valid: s = make_valid(s)
            if not s.is_empty and s.geom_type in ['Polygon', 'MultiPolygon']:
                geometries.append(s)
        except: continue
    return geometries

def main():
    # 1. Initialize Model
    model = SegFormer(NUM_CLASSES).to(device)
    if USE_TORCH_COMPILE and hasattr(torch, "compile"):
        model = torch.compile(model)

    ckpt = torch.load(MODEL_PATH, map_location=device)
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    state_dict = _adapt_state_dict_for_model(state_dict, model)
    model.load_state_dict(state_dict)
    model.eval()
    
    transform = EvalTransforms(size=(512, 512))
    post_processor = PostProcessing(NUM_CLASSES)

    input_file = input("Enter the .tiff file name: ").strip()
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    with rasterio.open(input_file) as src:
        H, W = src.height, src.width
        if src.count < 3:
            raise ValueError(f"Expected at least 3 channels, found {src.count}")
        
        # 2. Prediction Loop (Folder-based)
        rows = range(0, H, STRIDE)
        cols = range(0, W, STRIDE)
        
        current_mask_files = []
        pbar = tqdm(total=len(rows)*len(cols), desc="Processing Patches")
        for r in rows:
            for c in cols:
                win = Window.from_slices((r, r + PATCH_SIZE), (c, c + PATCH_SIZE))
                
                # --- A. Read and Save Input Patch ---
                patch = src.read(window=win, boundless=True, out_shape=(src.count, PATCH_SIZE, PATCH_SIZE))
                rgb_patch = patch[:3].transpose(1, 2, 0).astype(np.uint8) # HWC for PIL
                
                patch_id = f"R{r}_C{c}"
                PILImage.fromarray(rgb_patch).save(os.path.join(DATASET_DIR, f"{patch_id}.png"))
                
                # --- B. Transform & Predict ---
                # Keep original integer dtype so EvalTransforms can scale correctly.
                img_t = tv_tensors.Image(torch.from_numpy(patch[:3]))
                dummy_m = tv_tensors.Mask(torch.zeros((1, PATCH_SIZE, PATCH_SIZE)))
                img_t, _ = transform(img_t, dummy_m)
                
                with torch.no_grad():
                    preds = model(img_t.unsqueeze(0).to(device))
                    mask_512 = post_processor(preds)[0]
                    mask_1024 = torch.nn.functional.interpolate(
                        mask_512.unsqueeze(0).unsqueeze(0).float(),
                        size=(PATCH_SIZE, PATCH_SIZE), mode="nearest"
                    ).squeeze().cpu().numpy().astype(np.uint8)

                # Crop edge patches so we do not vectorize predictions outside raster bounds.
                valid_h = min(PATCH_SIZE, H - r)
                valid_w = min(PATCH_SIZE, W - c)
                mask_patch = mask_1024[:valid_h, :valid_w]

                # --- C. Save Predicted Mask ---
                mask_file = f"{patch_id}_mask.png"
                PILImage.fromarray(mask_patch).save(os.path.join(MASK_DIR, mask_file))
                current_mask_files.append(mask_file)
                pbar.update(1)
        pbar.close()

    # 3. Vectorization (Iterate through the saved masks in MASK_DIR)
    # To keep GPS accuracy, we re-calculate the transform based on the filename
    for class_val in [1, 2, 3]:
        logging.info(f"Vectorizing Class {class_val} from folder...")
        all_geoms = []
        
        tasks = []

        with rasterio.open(input_file) as src: # Re-open to get base transform
            for m_file in current_mask_files:
                # Parse row/col from filename "R1024_C0_mask.png"
                m = re.match(r"^R(\d+)_C(\d+)_mask\.png$", m_file)
                if m is None:
                    continue
                r_val, c_val = int(m.group(1)), int(m.group(2))
                
                mask_path = os.path.join(MASK_DIR, m_file)
                chunk = np.array(PILImage.open(mask_path))
                
                if np.any(chunk == class_val):
                    # Calculate geographic position of this specific patch
                    chunk_h, chunk_w = chunk.shape[:2]
                    chunk_win = Window.from_slices((r_val, r_val + chunk_h), (c_val, c_val + chunk_w))
                    chunk_trans = src.window_transform(chunk_win)
                    tasks.append((chunk, chunk_trans, class_val))

        if tasks:
            max_workers = max(1, (os.cpu_count() or 1) - 1)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(vectorize_chunk, t) for t in tasks]
                for f in as_completed(futures):
                    res = f.result()
                    if res: all_geoms.extend(res)

        if all_geoms:
            # Re-open original to get CRS
            with rasterio.open(input_file) as src:
                gdf = gpd.GeoDataFrame({'geometry': all_geoms}, crs=src.crs)
                output_shp = f"output_class_{class_val}.shp"
                gdf.to_file(output_shp)
                logging.info(f"Saved: {output_shp}")

if __name__ == "__main__":
    main()