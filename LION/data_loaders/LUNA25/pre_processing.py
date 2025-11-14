# This file is part of LION library
# License : BSD-3
#
# Author  : Andrea Sainz
# Modifications: -
# =============================================================================


import SimpleITK as sitk
import numpy as np
import glob
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import logging
import csv


# Paths
img_path = '/store/LION/as3628/datasets/LUNA25/raw/luna25_images/'
save_img_path = '/store/LION/as3628/datasets/LUNA25/preprocessed/luna25_images/'
mapping_csv_path = '/store/LION/as3628/datasets/LUNA25/preprocessed/patient_mapping.csv'

log_path = '/home/as3628/LION/LION/data_loaders/LUNA25/preprocessing.log'

os.makedirs(save_img_path, exist_ok=True)

# Background values
BACKGROUND_THRESHOLD = -1000
AIR_VALUE = -1000  # air HU value

# Take all images
image_files = sorted(glob.glob(os.path.join(img_path, "*.mha")))



def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Prevent adding handlers multiple times
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


logger = setup_logging()
logger.info(f"Found {len(image_files)} .mha files")


def process_one(args):
    i, file = args
    filename = os.path.basename(file)

    # Define patient_id and file names
    patient_id = f"patient_{i}"
    new_npy_filename = f"{patient_id}.npy"
    prefix = f"[{i+1}/{len(image_files)}] {patient_id} ({filename})"


    #check points
    logger.info(f"{prefix} - Starting processing")

    # read image
    img = sitk.ReadImage(file)

    # Metadata before editing
    spacing = img.GetSpacing()        # (sx, sy, sz)
    origin = img.GetOrigin()          # (ox, oy, oz)
    direction = img.GetDirection()    # 3×3 flattened

    # to NumPy (z, y, x)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)

    # Check original range
    orig_min = float(arr.min())
    orig_max = float(arr.max())
    logger.info(f"{prefix} - Original min/max HU: {orig_min:.1f} / {orig_max:.1f}")

    if orig_min > BACKGROUND_THRESHOLD:
        logger.warning(f"{prefix} - No voxels below {BACKGROUND_THRESHOLD} HU in {filename}")

    # background mask
    mask_background = arr < BACKGROUND_THRESHOLD

    # stats of background pixels (another check)
    n_bg_voxels = int(mask_background.sum())
    total_voxels = arr.size
    frac_bg = n_bg_voxels / total_voxels * 100
    logger.info(f"{prefix} - Background voxels (< {BACKGROUND_THRESHOLD}): "
                f"{n_bg_voxels} ({frac_bg:.3f}%)")
    
    # set background to air
    arr[mask_background] = AIR_VALUE

    # Check after modifying
    new_min = float(arr.min())
    new_max = float(arr.max())
    logger.info(f"{prefix} - Original min/max HU: {orig_min:.1f} / {orig_max:.1f}")
    logger.info(f"{prefix} - New min/max HU: {new_min:.1f} / {new_max:.1f}")

    if (arr < BACKGROUND_THRESHOLD).any():
       logger.error(f"{prefix} - WARNING: still found values below threshold after cleaning.")


    # Save NumPy version
    out_npy_path = os.path.join(save_img_path, new_npy_filename)
    np.save(out_npy_path, arr)
    logger.info(f"{prefix} -  Saved cleaned image to: {out_npy_path}")

    # plot first slice every 100 images
    if (i % 100) == 0:
        first_slice = arr[0]

        plt.figure(figsize=(5, 5))
        im =plt.imshow(first_slice, cmap="gray", origin="lower",
                   vmin=-1000, vmax=400)
        plt.axis("off")
        plt.colorbar(im)
        plt.title(f"Check slice 0 - {i}")
        plt.savefig(f"check_slice0_{i:04d}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"{prefix} - Saved check slice PNG: check_slice0_{i:04d}.png")

    return {
        "patient_id": patient_id,
        "original_filename": filename,

        # 3 ARRAYS IN THE CSV
        "dims": [img.GetSize()[0], img.GetSize()[1], img.GetSize()[2]],
        "spacing": [spacing[0], spacing[1], spacing[2]],
        "offset": [origin[0], origin[1], origin[2]],
        "direction": list(direction),   # 9 elements
    }



if __name__ == "__main__":
    n_workers = 20
    logger.info(f"Running with {n_workers} parallel workers (CPU).")

    # Collect results from workers
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(process_one, enumerate(image_files)))

    logger.info("FINISHED preprocessing all images.")
    logger.info(f"Writing patient mapping CSV to {mapping_csv_path}")

    # >>> NEW <<< Write CSV
    fieldnames = [
        "patient_id",
        "original_filename",
        "dims",
        "spacing",
        "offset",
        "direction",
    ]

    with open(mapping_csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            # Convert lists to strings for CSV
            for key in ["dims", "spacing", "offset", "direction"]:
                row[key] = str(row[key])
            writer.writerow(row)

    logger.info("Saved patient mapping CSV with metadata arrays.")