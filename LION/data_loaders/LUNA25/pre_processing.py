"""
This file is part of LION library
License : BSD-3

Author  : Andrea Sainz
Modifications: -



This script preprocesses the full LUNA25 dataset (CT volumes and nodule segmentations)
and produces a unified MASTER CSV linking all derived files.

What the script does:

1. Reads the LUNA25 annotation CSV and enriches it with:
   - patient_index (1…N)
   - study_index (per-patient CT scan index)
   - patient_name  (e.g. "patient_42_0")
   - num_scans_per_patient
   - num_nodules_per_scan
   - nodule_index (0,1,2,... within each CT)

2. For each patient (optionally in parallel with 20 workers):
   - Loads each CT scan (.mha)
   - Cleans background voxels (< -1000 HU → -1000 HU)
   - Saves the cleaned CT as a .npy file with a new structured name
   - For 10 random CT scans: saves a random slice PNG for debugging
   - Loads each nodule segmentation block (.npy)
   - Renames and copies the nodule image to a clean directory
   - Extracts and stores nodule metadata (shape, origin, spacing, transform) in memory
     instead of saving extra metadata .npy files.

3. After all patients are processed:
   - Merges original CSV info + CT metadata + nodule metadata
   - Writes a single MASTER CSV:
        LUNA25_preprocessed/LUNA25_master.csv

Output structure:
    preprocessed/
        luna25_images/               cleaned CT volumes (.npy)
        luna25_nodule_blocks/        renamed nodule crops (.npy)
        LUNA25_master.csv            one row per nodule with:
                                     - original LUNA25 fields
                                     - CT metadata (dims, spacing, origin, direction, paths)
                                     - nodule metadata (shape, origin, spacing, transform, name)
"""


import SimpleITK as sitk
import numpy as np
import glob
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import csv
import shutil 
import random
import pandas as pd

max_workers = 20


# PATHS
data_path = "/store/LION/as3628/datasets/LUNA25/raw/"
save_path = "/store/LION/as3628/datasets/LUNA25/preprocessed/"

img_path = os.path.join(data_path, "luna25_images/")
save_img_path = os.path.join(save_path, "luna25_images/")

seg_path = os.path.join(data_path, "luna25_nodule_blocks/")
seg_img_raw = os.path.join(seg_path, "image/")
seg_meta_raw = os.path.join(seg_path, "metadata/")

save_seg_img = os.path.join(save_path, "luna25_nodule_blocks/")

annotation_path = os.path.join(data_path, "LUNA25_Public_Training_Development_Data.csv")

log_path = "preprocessing.log"

# make sure the save path exist
os.makedirs(save_path, exist_ok=True)
os.makedirs(save_img_path, exist_ok=True)
os.makedirs(save_seg_img, exist_ok=True)



# constant background values
BACKGROUND_THRESHOLD = -1000
AIR_VALUE = -1000  # air HU value





# this defines a logger where we can store the information 
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

#start the logger
logger = setup_logging()



# read csv with all annotation and patien information
logger.info(f"Reading annotations CSV from: {annotation_path}")
ann_csv = pd.read_csv(annotation_path)


# create an index for each patient (1-N)
unique_patient_ids = sorted(ann_csv["PatientID"].unique())
patientid_to_index = {pid: i + 1 for i, pid in enumerate(unique_patient_ids)}

# identify if a patient has more than one CT in the database
series_to_indices = {}
for pid in unique_patient_ids:
    sub = ann_csv[ann_csv["PatientID"] == pid]

    # Unique scans for this patient (SeriesInstanceUID, StudyDate)
    scans = (
        sub[["SeriesInstanceUID", "StudyDate"]]
        .drop_duplicates()
        .sort_values(["StudyDate", "SeriesInstanceUID"])
    )

    p_idx = patientid_to_index[pid]

    for study_idx, row in enumerate(scans.itertuples()):
        series_uid = row.SeriesInstanceUID
        # thsi is for generating patient names in the format 
        #patient_X_E where X is the patient number and E is the scanner number for the patient
        series_to_indices[series_uid] = {
            "patient_index": p_idx,   # X
            "study_index": study_idx  # E
        }

# add the new columns to the dataframe row by row
ann_csv["patient_index"] = ann_csv["PatientID"].map(patientid_to_index)
ann_csv["study_index"] = ann_csv["SeriesInstanceUID"].map(
    lambda uid: series_to_indices[uid]["study_index"]
)

# build "patient_X_E" name
ann_csv["patient_name"] = (
    "patient_"
    + ann_csv["patient_index"].astype(str)
    + "_"
    + ann_csv["study_index"].astype(str)
)

# number of CT scans per patient
ann_csv["num_scans_per_patient"] = (
    ann_csv.groupby("PatientID")["SeriesInstanceUID"]
    .transform("nunique")
)

# number of nodules per scan (there are differetn nodule ID for different nodules in the csv)
ann_csv["num_nodules_per_scan"] = (
    ann_csv.groupby("SeriesInstanceUID")["LesionID"]
    .transform("nunique")
)

# Nodule index within each scan (0,1,2,...), for naming
ann_csv["nodule_index"] = (
    ann_csv.groupby("SeriesInstanceUID")
    .cumcount()
)


logger.info("Annotated CSV enriched with patient_index, study_index, patient_name, "
            "num_nodules_per_scan, nodule_index.")








# function for one patient (but maybe more than 1 CT and more than 1 nodule per CT)
def process_one_patient(patient_df: pd.DataFrame):
    # ct and nodule info
    ct_results = []        
    nodule_results = []    

    pid = patient_df["PatientID"].iloc[0]
    logger.info(f"Processing PatientID={pid} with {patient_df['SeriesInstanceUID'].nunique()} CT scans")


    # to save which Ct we already processed
    processed_series = set()


    for series_uid, scan_df in patient_df.groupby("SeriesInstanceUID"):
        patient_name = scan_df["patient_name"].iloc[0]
        p_idx = int(scan_df["patient_index"].iloc[0])
        study_idx = int(scan_df["study_index"].iloc[0])


        ct_mha_path = os.path.join(img_path, f"{series_uid}.mha")
        ct_npy_name = f"{patient_name}.npy"
        ct_npy_path = os.path.join(save_img_path, ct_npy_name)

        prefix = f"[PatientID={pid} | {patient_name}]"
        logger.info(f"{prefix} - Starting CT processing")

        if series_uid not in processed_series:
            if not os.path.exists(ct_mha_path):
                logger.error(f"{prefix} - CT file not found: {ct_mha_path}")
                continue

            # Read image
            img = sitk.ReadImage(ct_mha_path)

            # Metadata
            spacing = img.GetSpacing()
            origin = img.GetOrigin()
            direction = img.GetDirection()
            dims = img.GetSize()

            # Convert to NumPy (z, y, x)
            arr = sitk.GetArrayFromImage(img).astype(np.float32)

            # Original HU stats
            orig_min = float(arr.min())
            orig_max = float(arr.max())
            logger.info(f"{prefix} - Original min/max HU: {orig_min:.1f} / {orig_max:.1f}")

            if orig_min > BACKGROUND_THRESHOLD:
                logger.warning(f"{prefix} - No voxels below {BACKGROUND_THRESHOLD} HU")

            # Background mask and cleaning
            mask_background = arr < BACKGROUND_THRESHOLD
            n_bg_voxels = int(mask_background.sum())
            total_voxels = arr.size
            frac_bg = n_bg_voxels / total_voxels * 100.0
            logger.info(
                f"{prefix} - Background voxels (< {BACKGROUND_THRESHOLD}): "
                f"{n_bg_voxels} ({frac_bg:.3f}%)"
            )

            arr[mask_background] = AIR_VALUE

            # Stats after cleaning
            new_min = float(arr.min())
            new_max = float(arr.max())
            logger.info(f"{prefix} - New min/max HU: {new_min:.1f} / {new_max:.1f}")

            if (arr < BACKGROUND_THRESHOLD).any():
                logger.error(f"{prefix} - WARNING: still found values below threshold after cleaning.")


            # Save cleaned CT as .npy with new name (patient_X_E.npy) and delete original CT .mha to free space
            try:
                np.save(ct_npy_path, arr)
                logger.info(f"{prefix} - Saved cleaned CT to: {ct_npy_path}")
                try:
                    os.remove(ct_mha_path)
                    logger.info(f"{prefix} - Deleted original CT: {ct_mha_path}")
                except Exception as e:
                    logger.warning(
                        f"{prefix} - Could not delete original CT {ct_mha_path}: {e}")
            except Exception as e:
                    logger.warning(
                        f"{prefix} - Could not save cleaned CT {ct_npy_path}: {e}")


            # Save random slice only for selected CTs
            if series_uid in debug_series:
                slice_idx = random.randint(0, arr.shape[0] - 1)
                debug_slice = arr[slice_idx]

                plt.figure(figsize=(5, 5))
                im = plt.imshow(debug_slice, cmap="gray", origin="lower")
                plt.axis("off")
                plt.colorbar(im)
                plt.title(f"{patient_name} - random slice {slice_idx}")
                debug_png = os.path.join(save_path, f"debug_{patient_name}_slice{slice_idx}.png")
                plt.savefig(debug_png, dpi=150, bbox_inches="tight")
                plt.close()

                logger.info(f"{prefix} - Saved DEBUG random slice PNG: {debug_png}")

            # Save ct info
            ct_results.append(
                {
                    "patient_name": patient_name,
                    "patient_index": p_idx,
                    "study_index": study_idx,
                    "nlst_PatientID": pid,
                    "SeriesInstanceUID": series_uid,
                    "ct_npy_path": ct_npy_path,
                    "dims": dims,
                    "spacing": spacing,
                    "offset": origin,
                    "direction": list(direction),
                }
            )

            processed_series.add(series_uid)

        # nodules files 
        logger.info(f"{prefix} - Processing {len(scan_df)} nodules")

        for row in scan_df.itertuples():
            annot_id = row.AnnotationID
            nodule_idx = int(row.nodule_index)
            nodule_label = int(row.label)
            lesion_id = int(row.LesionID)

            new_base = f"{patient_name}_nodule_{nodule_idx}"

            src_img_npy = os.path.join(seg_img_raw, f"{annot_id}.npy")
            src_meta_npy = os.path.join(seg_meta_raw, f"{annot_id}.npy")

            dst_img_npy = os.path.join(save_seg_img, f"{new_base}.npy")

            # nodule segmentation image
            if not os.path.exists(src_img_npy):
                logger.error(f"{prefix} - Nodule image .npy not found: {src_img_npy}")
                continue

            # load and move segmentation image
            nodule_img = np.load(src_img_npy)
            try:
                os.replace(src_img_npy, dst_img_npy) 
                logger.info(f"{prefix} - Moved nodule image to: {dst_img_npy}")
            except Exception as e:
                logger.warning(f"{prefix} - Could not move nodule image {src_img_npy} -> {dst_img_npy}: {e}")
                continue 

            # nodule segmentation metadata (only extract infor, no saving)
            if not os.path.exists(src_meta_npy):
                logger.error(f"{prefix} - Nodule metadata .npy not found: {src_meta_npy}")
                continue

            meta = np.load(src_meta_npy, allow_pickle=True).item()
            seg_origin = meta.get("origin", None)
            seg_spacing = meta.get("spacing", None)
            seg_transform = meta.get("transform", None)

            logger.info(f"{prefix} - Extracted nodule metadata from: {src_meta_npy}")

            #remove numpy file with metadata
            try:
                os.remove(src_meta_npy)
                logger.info(f"{prefix} - Deleted metadata numpy file: {src_meta_npy}")
            except Exception as e:
                logger.warning(f"{prefix} - Could not delete metadata numpy file {src_meta_npy}: {e}")

            # Save information about the nodule
            nodule_name = f"{patient_name}_nodule_{nodule_idx}"
            seg_shape = tuple(nodule_img.shape)

            nodule_results.append(
            {
                # patient info
                "patient_name": patient_name,
                "patient_index": p_idx,
                "study_index": study_idx,
                "nlst_PatientID": pid,

                # scan info
                "SeriesInstanceUID": series_uid,

                # nodule info
                "nodule_name": nodule_name,
                "LesionID": lesion_id,
                "AnnotationID": annot_id,
                "nodule_index": nodule_idx,
                "label": nodule_label,

                # paths
                "nodule_img_name": new_base,

                # segmentation metadata
                "seg_shape": str(seg_shape),
                "seg_origin": str(seg_origin),
                "seg_spacing": str(seg_spacing),
                "seg_transform": str(seg_transform),
            }
            )
    return ct_results, nodule_results














# we choose 10 random CT from the whole list to take a random slice 
all_series_ids = ann_csv["SeriesInstanceUID"].unique().tolist()
debug_series = set(random.sample(all_series_ids, 10))

logger.info(f"Debugging enabled: will save random slices for {len(debug_series)} CT scans.")


if __name__ == "__main__":
    logger.info("Starting LUNA25 preprocessing (CT and nodules)")

    # Convert grouped patients into a list of dataframes (each for one patient)
    patient_groups = [
        patient_df
        for _, patient_df in ann_csv.groupby("PatientID")
    ]

    all_ct_results = []
    all_nodule_results = []

    # Parallel processing using 20 CPUs
    logger.info(f"Launching parallel preprocessing using {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_one_patient, df): df["PatientID"].iloc[0]
            for df in patient_groups
        }

        for future in as_completed(futures):
            pid = futures[future]
            try:
                ct_res, nodule_res = future.result()
                all_ct_results.extend(ct_res)
                all_nodule_results.extend(nodule_res)
                logger.info(f"[PatientID={pid}] finished successfully.")
            except Exception as e:
                logger.error(f"[PatientID={pid}] failed with: {e}")

    logger.info("Finished parallel preprocessing of all patients.")

    logger.info("Writing final MASTER CSV")

    master_csv_path = os.path.join(save_path, "LUNA25_master.csv")

    # lookup de CT por SeriesInstanceUID
    ct_lookup = {
        row["SeriesInstanceUID"]: row
        for row in all_ct_results
    }

    # lookup de fila original del CSV por AnnotationID
    ann_lookup = (
        ann_csv
        .set_index("AnnotationID")
        .to_dict(orient="index")
    )

    master_rows = []

    for nrow in all_nodule_results:
        series_uid = nrow["SeriesInstanceUID"]
        annot_id = nrow["AnnotationID"]

        ct_info = ct_lookup[series_uid]
        # this takes all the original columns
        orig = ann_lookup[annot_id]  

        # create a dictionary with all the original info
        row_dict = dict(orig)

        # adding all the information per patient,ct and nodule
        row_dict.update({
            # new names used
            "patient_name": nrow["patient_name"],
            "patient_index": nrow["patient_index"],
            "study_index": nrow["study_index"],
            "nodule_name": nrow["nodule_name"],

            # info CT
            "ct_mha_path": ct_info["ct_mha_path"],
            "ct_npy_path": ct_info["ct_npy_path"],
            "ct_dims": str(ct_info["dims"]),
            "ct_spacing": str(ct_info["spacing"]),
            "ct_offset": str(ct_info["offset"]),
            "ct_direction": str(ct_info["direction"]),

            # info nodule
            "nodule_index": nrow["nodule_index"],
            "label": nrow["label"],
            "nodule_img_name": nrow["nodule_img_name"],
            "seg_shape": nrow["seg_shape"],
            "seg_origin": nrow["seg_origin"],
            "seg_spacing": nrow["seg_spacing"],
            "seg_transform": nrow["seg_transform"],
        })

        master_rows.append(row_dict)

    # write csv with all the information
    fieldnames = list(master_rows[0].keys())

    with open(master_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in master_rows:
            writer.writerow(row)

    logger.info(f"Saved MASTER CSV to {master_csv_path}")