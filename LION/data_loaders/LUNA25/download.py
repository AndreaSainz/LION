""" 
This file is part of LION library
License : BSD-3

Author  : Andrea Sainz Bear



This script downloads the PUBLIC LUNA25 dataset (CT scans + nodule annotations)
from Zenodo and prepares it inside the LION dataset directory structure.

It supports *selective* downloading:
- Only the CT scans (multi-part .zip split archives)
- Only the annotation CSV files
- Both CT scans + annotation metadata
- Optional automatic extraction of the CT split archives via 7-Zip

--------------------------------------------------------------------------------
DATA SOURCES
--------------------------------------------------------------------------------
The script downloads two Zenodo records:

1) IMAGING RECORD  (IMAGING_RECORD_ID)
   Contains:
     - 4069 low-dose CT scans (multi-part split .zip files)
     - extracted into: <outdir>/luna25_images/
                       <outdir>/luna25_nodule_blocks/

2) ANNOTATION RECORD  (ANNOT_RECORD_ID)
   Contains:
     - CSV files with metadata and malignancy labels
     - stored directly in: <outdir>/

A valid `ZENODO_TOKEN` must be exported before running:
    export ZENODO_TOKEN='your_personal_access_token'

--------------------------------------------------------------------------------
USER OPTIONS
--------------------------------------------------------------------------------
Set the following flags at the top of the script:

    DOWNLOAD_CT = True / False
        - If True, downloads the CT scans (split .zip files).
        - If False, skips all imaging files.

    DOWNLOAD_ANNOTATIONS = True / False
        - If True, downloads the annotation CSV files.
        - If False, skips all annotation files.

    EXTRACT_CT = True / False
        - If True, automatically extracts the CT multi-part archives using 7-Zip.
        - Ignored if DOWNLOAD_CT=False.

    CLEAN_ZIPS_AFTER_EXTRACT = True / False
        - If True, deletes the downloaded .zip.001/.002/... parts after extraction.

--------------------------------------------------------------------------------
DEPENDENCIES
--------------------------------------------------------------------------------
Requires:
- A valid Zenodo API token (see above).
- p7zip installed (for extraction):
      conda install -c conda-forge p7zip
- Network access
- Sufficient disk space (~250GB for full CT dataset)

--------------------------------------------------------------------------------
OUTPUT DIRECTORY
--------------------------------------------------------------------------------
All downloaded and extracted files are stored in:

    /store/LION/as3628/datasets/LUNA25/raw/

Modify `outdir` at the top of the script if needed.

--------------------------------------------------------------------------------
BEHAVIOR SUMMARY
--------------------------------------------------------------------------------
- Parallel downloads (ThreadPoolExecutor)
- Integrity checks: skip existing complete files
- Clean extraction pipeline
- Optional cleanup of downloaded zip chunks
"""


import os
import pathlib
import requests
import subprocess        
import shutil
from LION.utils.paths import LUNA25_DATASET_PATH
from concurrent.futures import ThreadPoolExecutor, as_completed

# OPTIONS: what to download and clean
DOWNLOAD_CT = True
DOWNLOAD_ANNOTATIONS = True
EXTRACT_CT = True
CLEAN_ZIPS_AFTER_EXTRACT = True

# === CONFIG ===
# create a Zenodo Token to access the data and then run in the terminal: export ZENODO_TOKEN='your_token' 
TOKEN = os.getenv("ZENODO_TOKEN")
# Imaging data (CT + blocks)
IMAGING_RECORD_ID = "14223624"
# Annotation data (CSV, etc.)
ANNOT_RECORD_ID = "14673658"

outdir = pathlib.Path("/store/LION/as3628/datasets/LUNA25/raw") # Change to LUNA25_DATASET_PATH in the future, I don't have permission
outdir.mkdir(parents=True, exist_ok=True)
max_workers = 20


# The two multi-part archives in LUNA25
extract_targets = [
    "luna25_images.zip.001",
    "luna25_nodule_blocks.zip.001",
]



# === DOWNLOAD FUNCTION ===
def download_one(fi, index, total_files):
    name = fi["key"]
    url = fi["links"]["self"]
    dst = outdir / name
    size = fi["size"]

    # skip if already exists and complete
    if dst.exists() and dst.stat().st_size == size:
        return f"[{index}/{total_files}] SKIP: {name} (already complete)"

    # download
    with requests.get(url, params={"access_token": TOKEN}, stream=True) as resp:
        resp.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in resp.iter_content(1024 * 1024):
                f.write(chunk)

    return f"[{index}/{total_files}] DONE: {name}"

# === GET CT FILE LIST ===
if DOWNLOAD_CT:
    print("=== CT SCANS ===")
    r = requests.get(f"https://zenodo.org/api/records/{IMAGING_RECORD_ID}", params={'access_token': TOKEN})
    r.raise_for_status()
    files = r.json()["files"]
    total_files = len(files)

    print(f"Total files to process: {total_files}")
    print(f"Output directory: {outdir}")

    # === PARALLEL CT DOWNLOADS ===
    completed = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(download_one, fi, i+1, total_files): fi for i, fi in enumerate(files)}
        for fut in as_completed(futures):
            msg = fut.result()
            print(msg)
            if "DONE" in msg:
                completed += 1
            elif "SKIP" in msg:
                skipped += 1

    # === SUMMARY ===
    print("\n Download summary (IMAGING):")
    print(f"  Completed: {completed}")
    print(f"  Skipped (already present): {skipped}")
    print(f"  Total processed: {completed + skipped}/{total_files}")
    print("\n Imaging downloads finished successfully")


if DOWNLOAD_ANNOTATIONS:
    # === GET SEGMENTATION FILE LIST ===
    print("=== ANNOTATIONS ===")
    r_seg = requests.get(f"https://zenodo.org/api/records/{ANNOT_RECORD_ID}", params={'access_token': TOKEN})
    r_seg.raise_for_status()
    files_seg = r_seg.json()["files"]
    total_files_seg = len(files_seg)

    print(f"Total files to process: {total_files_seg}")
    print(f"Output directory: {outdir}")

    # === PARALLEL SEGMENTATION DOWNLOADS ===
    completed_seg = 0
    skipped_seg = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(download_one, fi, i+1, total_files_seg): fi for i, fi in enumerate(files_seg)}
        for fut in as_completed(futures):
            msg = fut.result()
            print(msg)
            if "DONE" in msg:
                completed_seg += 1
            elif "SKIP" in msg:
                skipped_seg += 1

    # === SUMMARY ===
    print("\n Download summary (ANNOTATIONS):")
    print(f"  Completed: {completed_seg}")
    print(f"  Skipped (already present): {skipped_seg}")
    print(f"  Total processed: {completed_seg + skipped_seg}/{total_files_seg}")
    print("\n Annotation downloads finished successfully")



# === EXTRACTION WITH 7-ZIP (SPLIT ARCHIVES) ===
def ensure_7z_available():
    if not shutil.which("7z"):
        raise RuntimeError("7z not found. Install p7zip (e.g., conda install -c conda-forge p7zip)")

def extract_split_archive(first_part: pathlib.Path):
    ensure_7z_available()
    cmd = ["7z", "x", str(first_part.resolve()), f"-o{outdir.resolve()}"]
    print(f"\n[EXTRACT] Running: {' '.join(cmd)}")

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)

    ok = "Everything is Ok" in proc.stdout
    if not ok:
        print(f"[EXTRACT] Warning: 7z did not report 'Everything is Ok' for {first_part.name}")
    return ok

def delete_zip_parts(prefix: str):
    """
    Deletes all split zip parts for a given prefix, e.g.:
    prefix = "luna25_images" → luna25_images.zip.001, .002, ...
    """
    count = 0
    # Esto pillará .zip.001, .zip.002, etc. (y también .zip si existiera)
    for p in outdir.glob(f"{prefix}.zip*"):
        try:
            p.unlink()
            count += 1
        except Exception as e:
            print(f"[CLEAN] Could not remove {p.name}: {e}")
    print(f"[CLEAN] Removed {count} zip part(s) matching '{prefix}.zip*'")

# === MAIN EXTRACTION LOOP ===
if EXTRACT_CT and DOWNLOAD_CT:
    for first_part_name in extract_targets:
        first_part_path = outdir / first_part_name
        if first_part_path.exists():
            ok = extract_split_archive(first_part_path)
            if ok and CLEAN_ZIPS_AFTER_EXTRACT:
                prefix = first_part_name.split(".zip.001")[0]
                delete_zip_parts(prefix)
        else:
            print(f"[EXTRACT] Skip: {first_part_name} not found (maybe not downloaded)")

print("\nAll tasks finished.")