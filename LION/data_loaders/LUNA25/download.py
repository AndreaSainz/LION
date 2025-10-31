# This file is part of LION library
# License : BSD-3
#
# Author  : Andrea Sainz Bear
# =============================================================================

import os
import pathlib
import requests
import subprocess        
import shutil
from LION.utils.paths import LUNA25_DATASET_PATH
from concurrent.futures import ThreadPoolExecutor, as_completed

# === CONFIG ===
# create a Zenodo Token to access the data and then run in the terminal: export ZENODO_TOKEN='your_token' 
TOKEN = os.getenv("ZENODO_TOKEN")
record_id = "14223624"
outdir = pathlib.Path("/store/LION/as3628/datasets/raw/LUNA25") # Change to LUNA25_DATASET_PATH in the future, I don't have permission
outdir.mkdir(parents=True, exist_ok=True)
max_workers = 12

# extra options
EXTRACT_AFTER_DOWNLOAD = True
CLEAN_ZIPS_AFTER_EXTRACT = True
# The two multi-part archives in LUNA25
extract_targets = [
    "luna25_images.zip.001",
    "luna25_nodule_blocks.zip.001",
]

# === GET FILE LIST ===
r = requests.get(f"https://zenodo.org/api/records/{record_id}", params={'access_token': TOKEN})
r.raise_for_status()
files = r.json()["files"]
total_files = len(files)

print(f"Total files to process: {total_files}")
print(f"Output directory: {outdir}")

# === DOWNLOAD FUNCTION ===
def download_one(fi, index):
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

# === PARALLEL DOWNLOADS ===
completed = 0
skipped = 0

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    futures = {ex.submit(download_one, fi, i+1): fi for i, fi in enumerate(files)}
    for fut in as_completed(futures):
        msg = fut.result()
        print(msg)
        if "DONE" in msg:
            completed += 1
        elif "SKIP" in msg:
            skipped += 1

# === SUMMARY ===
print("\n Download summary:")
print(f"  Completed: {completed}")
print(f"  Skipped (already present): {skipped}")
print(f"  Total processed: {completed + skipped}/{total_files}")
print("\n All downloads finished successfully")


# === EXTRACTION WITH 7-ZIP (SPLIT ARCHIVES) ===
def ensure_7z_available():
    if not shutil.which("7z"):
        raise RuntimeError("7z not found. Install p7zip (e.g., conda install -c conda-forge p7zip)")

def extract_split_archive(first_part: pathlib.Path):
    """
    Extracts a split archive (.zip.001, .zip.002, ...) in the current directory.
    Returns True if 'Everything is Ok' appears in the output.
    """
    ensure_7z_available()
    cmd = ["7z", "x", str(first_part), "-o."]
    print(f"\n[EXTRACT] Running: {' '.join(cmd)}")

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)

    ok = "Everything is Ok" in proc.stdout
    if not ok:
        print(f"[EXTRACT] Warning: 7z did not report 'Everything is Ok' for {first_part.name}")
    return ok

def delete_zip_parts(prefix: str):
    """
    Deletes all files that start with a given prefix and contain '.zip'
    (e.g., .zip.001, .zip.002, ...).
    """
    count = 0
    for p in outdir.glob(f"{prefix}*.zip*"):
        try:
            p.unlink()
            count += 1
        except Exception as e:
            print(f"[CLEAN] Could not remove {p.name}: {e}")
    print(f"[CLEAN] Removed {count} zip part(s) matching '{prefix}*.zip*'")

# === MAIN EXTRACTION LOOP ===
if EXTRACT_AFTER_DOWNLOAD:
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