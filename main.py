"""This script implements a complete pipeline for processing large geospatial .tiff images using a SegFormer"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import geopandas as gpd
import logging
from model import SegFormer
import numpy as np
import os
from PIL import Image as PILImage
import re
import rasterio
from rasterio.windows import Window
from rasterio import features
import torch
from torchvision import tv_tensors
from tqdm import tqdm
from transforms import EvalTransforms, PostProcessing
import shutil
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, shape
from shapely.ops import unary_union
from shapely.validation import make_valid
import signal
import sys
from utils import device_setup, setup_logging, handle_shutdown, shutdown_requested

pin_memory = False
amp_dtype = torch.bfloat16
logger = logging.getLogger(__name__)

PILImage.MAX_IMAGE_PIXELS = None

#Configuration
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


def _is_blank_patch(patch_rgb, nodata):
    """Return True for empty/no-data patches to avoid model edge artifacts."""
    patch = np.asarray(patch_rgb, dtype=np.float32)
    if patch.size == 0:
        return True
    if np.isnan(patch).all():
        return True

    finite = patch[np.isfinite(patch)]
    if finite.size == 0:
        return True

    # Nearly constant tiles are usually nodata fill in orthophoto boundaries.
    if (float(np.max(finite)) - float(np.min(finite))) <= 1.0:
        return True

    # Heuristic for almost-all black/white tiles after boundless reads.
    black_ratio = float(np.mean(patch <= 1.0))
    white_ratio = float(np.mean(patch >= 254.0))
    if black_ratio >= 0.995 or white_ratio >= 0.995:
        return True

    if nodata is not None:
        nodata_ratio = float(np.mean(np.isclose(patch, float(nodata), atol=1.0)))
        if nodata_ratio >= 0.995:
            return True

    return False

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
                if shutdown_requested:
                    sys.exit(0)
                win = Window.from_slices((r, r + PATCH_SIZE), (c, c + PATCH_SIZE))
                
                #Exytract the .tiff patch
                patch = src.read(window=win, boundless=True, out_shape=(src.count, PATCH_SIZE, PATCH_SIZE))
                rgb_patch = patch[:3].transpose(1, 2, 0).astype(np.uint8) # HWC for PIL
                
                patch_id = f"R{r}_C{c}"
                PILImage.fromarray(rgb_patch).save(os.path.join(DATASET_DIR, f"{patch_id}.png"))


                patch_rgb_chw = patch[:3]
                if _is_blank_patch(patch_rgb_chw, src.nodata):
                    mask_1024 = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
                else:
                    # Making a dummy mask to be passed into EvalTransform as it needs both an image and a mask.
                    img_t = tv_tensors.Image(torch.from_numpy(patch_rgb_chw))
                    dummy_m = tv_tensors.Mask(torch.zeros((1, PATCH_SIZE, PATCH_SIZE)))
                    img_t, _ = transform(img_t, dummy_m)

                    # Getting the prediction mask and resizing it up.
                    with torch.no_grad():
                        preds = model(img_t.unsqueeze(0).to(device))
                        mask_512 = post_processor(preds)[0]
                        mask_1024 = torch.nn.functional.interpolate(
                            mask_512.unsqueeze(0).unsqueeze(0).float(),
                            size=(PATCH_SIZE, PATCH_SIZE), mode="nearest"
                        ).squeeze().cpu().numpy().astype(np.uint8)

                #Cropping edges so it does not rasterize outside the boundaries
                valid_h = min(PATCH_SIZE, H - r)
                valid_w = min(PATCH_SIZE, W - c)
                mask_patch = mask_1024[:valid_h, :valid_w]

                # Saved the predicted mask
                mask_file = f"{patch_id}_mask.png"
                PILImage.fromarray(mask_patch).save(os.path.join(MASK_DIR, mask_file))
                current_mask_files.append(mask_file)
                pbar.update(1)
        pbar.close()

    #Iterates throughs all the masks and then merges them into an .shp vector file
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
                # Dissolve touching polygons so tile boundaries are removed.
                merged = unary_union(all_geoms)
                if merged.is_empty:
                    continue

                if isinstance(merged, Polygon):
                    merged_geoms = [merged]
                elif isinstance(merged, MultiPolygon):
                    merged_geoms = list(merged.geoms)
                elif isinstance(merged, GeometryCollection):
                    merged_geoms = [g for g in merged.geoms if isinstance(g, (Polygon, MultiPolygon))]
                else:
                    merged_geoms = []

                if not merged_geoms:
                    logging.info(f"No polygon geometry remained after merge for class {class_val}")
                    continue

                gdf = gpd.GeoDataFrame({'geometry': merged_geoms}, crs=src.crs)

                #Renaming the aquired .shps so the names batch the classes inside them.
                if class_val == 1:
                    output_shp = "Road.shp"
                elif class_val == 2:
                    output_shp = "BuildUpArea.shp"
                else:
                    output_shp = "WaterBodies.shp"

                gdf.to_file(output_shp)
                logging.info(f"Saved: {output_shp}")


    # Delete temporary inference folders created during this run.
    for folder in [MASK_DIR, DATASET_DIR]:
        if os.path.isdir(folder):
            try:
                shutil.rmtree(folder)
                logging.info(f"Deleted temporary folder: {folder}")
            except Exception as exc:
                logging.warning(f"Could not delete folder '{folder}': {exc}")

if __name__ == "__main__":

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    device, pin_memory, amp_dtype = device_setup()

    setup_logging()

    main()