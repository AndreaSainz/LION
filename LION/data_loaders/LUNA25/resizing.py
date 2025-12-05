import SimpleITK as sitk
import numpy as np
import pandas as pd
from tqdm import tqdm 
import os
import logging
import random
import matplotlib
from pathlib import Path
matplotlib.use("Agg")  # we only want to save PNGs, no GUI backend needed
import matplotlib.pyplot as plt


csv_path = "/store/LION/as3628/nosnap/LUNA25/preprocessed/LUNA25_master.csv"
save_path = "/store/LION/as3628/nosnap/LUNA25/preprocessed/luna25_images_uniform"
os.makedirs(save_path, exist_ok=True)
# log file path
log_path = str(Path.home() / "LION/LION/data_loaders/LUNA25/resample_luna25.log")

# this defines a logger where we can store the information 
def setup_logging():
    logger = logging.getLogger("luna25_resample")
    logger.setLevel(logging.INFO)

    # Prevent adding handlers multiple times
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# start the logger
logger = setup_logging()

logger.info("==== Starting LUNA25 in-plane resampling to 0.7mm / 512x512 ====")

# read csv with all information
df = pd.read_csv(csv_path)

# delete same CT files (1 row per nodule, not per CT)
unique_cts = df.drop_duplicates(subset="ct_npy_path")[["ct_npy_path", "ct_dims", "ct_spacing"]]

# Convert text tu tuples
unique_cts["ct_dims"] = unique_cts["ct_dims"].apply(eval)
unique_cts["ct_spacing"] = unique_cts["ct_spacing"].apply(eval)

logger.info(f"Found {len(unique_cts)} unique CT volumes from LUNA25_master CSV.")


# choose 10 random CTs for saving debug slices
all_ct_paths = unique_cts["ct_npy_path"].unique().tolist()
n_debug = min(10, len(all_ct_paths))
debug_ct_paths = set(random.sample(all_ct_paths, n_debug))
logger.info(f"Selected {n_debug} CTs for debug slice PNGs.")


# resampling function
def resample_to_spacing_keep_size(
    slice_arr, 
    spacing_xy,
    target_spacing=(0.7, 0.7), 
    output_size=(512, 512),
    interpolator=sitk.sitkLinear
):
    
    # convert imag to ITK
    img = sitk.GetImageFromArray(slice_arr.astype(np.float32))  # (y, x) -> size (x, y)
    img.SetSpacing((float(spacing_xy[0]), float(spacing_xy[1])))
    orig_size = np.array(img.GetSize(), dtype=np.float32) # (nx, ny)


    # physic resample to target_spacing
    scale = spacing_xy / np.array(target_spacing, dtype=np.float32)
    new_size = (orig_size * scale).astype(int).tolist()

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetInterpolator(interpolator)
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetOutputDirection(img.GetDirection())

    img_resampled = resample.Execute(img)

    # crop/padding for output size
    resample2 = sitk.ResampleImageFilter()
    resample2.SetOutputSpacing(target_spacing)
    resample2.SetSize(output_size)
    resample2.SetInterpolator(interpolator)
    final_img = resample2.Execute(img_resampled)

    return sitk.GetArrayFromImage(final_img)


target_spacing = (0.7, 0.7)
output_size = (512, 512)

for _, row in tqdm(unique_cts.iterrows(), total=len(unique_cts)):
    ct_path = row["ct_npy_path"]
    spacing_orig = row["ct_spacing"]  # (sx, sy, sz)
    size_orig = row["ct_dims"]        # (nx, ny, nz)
    nx, ny, nz = size_orig
    dims_numpy = (nz, ny, nx)

    base = os.path.basename(ct_path)
    prefix = f"[CT={base}]"

    logger.info(
        f"{prefix} Starting resampling. "
        f"Original spacing={spacing_orig}, dims={dims_numpy}, npy={ct_path}"
    )

    # we only care about sx,sy for now
    spacing_xy = np.array(spacing_orig[:2], dtype=np.float32)

    ct_scan = np.load(ct_path)        # (z, y, x)
    nz, ny, nx = ct_scan.shape
    if (nx != size_orig[0]) or (ny != size_orig[1]) or (nz != size_orig[2]):
        msg = (
            f"{prefix} Shape of .npy {ct_scan.shape} does not match "
            f"ct_dims from csv {size_orig}"
        )
        logger.error(msg)
        raise ValueError(msg)



    # output size
    resampled = np.empty((nz, output_size[1], output_size[0]), dtype=np.float32)

    for z in range(nz):
        slice_arr = ct_scan[z, :, :]
        resampled[z, :, :] = resample_to_spacing_keep_size(
            slice_arr,
            spacing_xy=spacing_xy,
            target_spacing=target_spacing,
            output_size=output_size,
            interpolator=sitk.sitkLinear,
        )

    # save new CTs (same filename, new folder)
    out_path = os.path.join(save_path, base)
    np.save(out_path, resampled)
    logger.info(
        f"{prefix} Finished resampling. Saved to {out_path} "
        f"with shape={resampled.shape}, target_spacing={target_spacing}."
    )

    # Save one random slice for selected CTs for visual debugging
    if ct_path in debug_ct_paths:
        slice_idx = random.randint(0, nz - 1)
        debug_slice = resampled[slice_idx]

        plt.figure(figsize=(5, 5))
        im = plt.imshow(debug_slice, cmap="gray", origin="lower")
        plt.axis("off")
        plt.colorbar(im)
        plt.title(f"{base} - resampled slice {slice_idx}")
        debug_png = os.path.join(
            save_path,
            f"debug_resampled_{os.path.splitext(base)[0]}_slice{slice_idx}.png"
        )
        plt.savefig(debug_png, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"{prefix} Saved DEBUG resampled slice PNG: {debug_png}")

logger.info("==== Finished LUNA25 in-plane resampling for all CT volumes ====")