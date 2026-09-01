from radiomics import (
    featureextractor,
    firstorder,
    shape,
    shape2D,
    glcm,
    glszm,
    glrlm,
    ngtdm,
    gldm,
)
import SimpleITK as sitk
from LION.data_loaders.LIDC_IDRI import LIDC_IDRI
import LION.experiments.ct_experiments as ct_experiments
import pathlib
import torch
from torch.utils.data import DataLoader, Subset
from collections import Counter
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import numpy as np
import seaborn as sns
import logging
import warnings

save_fig_path = "scripts/pipeline_project2"

### If we want to change doses we can change the experiment completely
# Define experiment
experiment = ct_experiments.NormalDoseCTSegmentation(dataset="LIDC-IDRI")

# %% Dataset
lidc_dataset = experiment.get_training_dataset()

#%% Define DataLoader
batch_size = 1
lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=False)
lidc_test = DataLoader(experiment.get_testing_dataset(), batch_size, True)
lidc_val = DataLoader(experiment.get_validation_dataset(), batch_size, True)


# Instantiate the extractor
logging.getLogger("radiomics").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", message="Shape features are only available 3D input")

warnings.filterwarnings("ignore", message="GLCM is symmetrical")
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.enableFeatureClassByName("shape2D")

# Make an array of the values
all_features = []
no_nodule_samples = []
invalid_mask_samples = []

for batch_id, batch in enumerate(lidc_dataloader):
    sinogram = batch[0]
    mask_tensor = batch[1]
    reconstruction_tensor = batch[2]
    diagnosis = batch[3]

    image_np = reconstruction_tensor[0, 0].cpu().numpy()
    mask_np = mask_tensor[0, 1].cpu().numpy().astype("uint8")

    # Check if mask has the expected dimensionality and shape
    if mask_np.ndim != 2 or mask_np.shape != image_np.shape:
        print(
            f"Invalid sample {batch_id}: "
            f"image={image_np.shape}, mask={mask_np.shape}"
        )

        invalid_mask_samples.append(
            {
                "sample_id": batch_id,
                "image_shape": str(image_np.shape),
                "mask_shape": str(mask_np.shape),
                "mask_ndim": mask_np.ndim,
                "reason": "invalid_mask_dimension_or_shape",
            }
        )

        continue

    # Check if there is a nodule
    if mask_np.sum() == 0:
        print(f"No nodule found in sample {batch_id}")

        no_nodule_samples.append(
            {
                "sample_id": batch_id,
                "image_shape": str(image_np.shape),
                "mask_shape": str(mask_np.shape),
                "reason": "no_nodule",
            }
        )

        continue

    # Check if mask is only a horizontal or vertical line
    rows, cols = np.where(mask_np > 0)

    is_horizontal_line = len(np.unique(rows)) == 1
    is_vertical_line = len(np.unique(cols)) == 1

    if is_horizontal_line or is_vertical_line:

        if is_horizontal_line:
            reason = "mask_is_horizontal_line"
        else:
            reason = "mask_is_vertical_line"

        print(f"Invalid sample {batch_id}: {reason} " f"(mask pixels={len(rows)})")

        invalid_mask_samples.append(
            {
                "sample_id": batch_id,
                "image_shape": str(image_np.shape),
                "mask_shape": str(mask_np.shape),
                "mask_ndim": mask_np.ndim,
                "reason": reason,
            }
        )

        continue

    image_sitk = sitk.GetImageFromArray(image_np)
    mask_sitk = sitk.GetImageFromArray(mask_np)

    assert mask_np.ndim == 2
    assert mask_np.shape == image_np.shape
    assert mask_sitk.GetDimension() == 2

    result = extractor.execute(image_sitk, mask_sitk)

    # quitar información de diagnóstico
    features = {k: v for k, v in result.items() if not k.startswith("diagnostics")}

    # convertir numpy arrays a escalares
    features = {
        k: float(v.item()) if isinstance(v, np.ndarray) else v
        for k, v in features.items()
    }

    # añadir información extra
    features["sample_id"] = batch_id
    features["diagnosis"] = diagnosis.item()

    all_features.append(features)


radiomics_df = pd.DataFrame(all_features)

# DataFrames of eliminated samples
invalid_mask_df = pd.DataFrame(invalid_mask_samples)
no_nodule_df = pd.DataFrame(no_nodule_samples)

print("\nTotal samples without nodule:", len(no_nodule_df))
print("Samples without nodule:", no_nodule_df["sample_id"].tolist())

print("\nTotal samples with invalid mask:", len(invalid_mask_df))
print("Samples with invalid mask:", invalid_mask_df["sample_id"].tolist())

print("\nTotal extracted radiomics samples:", len(radiomics_df))


# Features to visualize
main_features = [
    "original_firstorder_Mean",
    "original_firstorder_Entropy",
    "original_firstorder_Kurtosis",
    "original_firstorder_Variance",
    "original_glcm_Contrast",
    "original_glcm_Correlation",
    "original_glcm_Entropy",
    "original_glrlm_RunEntropy",
    "original_glszm_ZoneEntropy",
    "original_gldm_DependenceEntropy",
]

for feature in main_features:

    plt.figure(figsize=(7, 4))

    plt.hist(radiomics_df[feature].dropna(), bins=30)

    plt.xlabel(feature)
    plt.ylabel("Number of samples")
    plt.title(f"Distribution of {feature}")

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_fig_path}_{feature}.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4))

    sns.histplot(
        data=radiomics_df,
        x=feature,
        hue="diagnosis",
        bins=30,
        kde=True,
        stat="density",
        common_norm=False,
    )

    plt.title(f"{feature}: benign vs malignant")
    plt.xlabel(feature)
    plt.ylabel("Density")

    plt.tight_layout()
    plt.savefig(f"{save_fig_path}_bvsm_{feature}.png", dpi=300)
    plt.close()


"""
Total samples without nodule: 126
Samples without nodule: [141, 152, 158, 488, 1283, 1730, 1880, 1881, 2120, 2272, 2417, 2533, 2854, 2868, 
3036, 3335, 3526, 3697, 3825, 3995, 3996, 4030, 4206, 4207, 4213, 4231, 4483, 4544, 4610, 4653, 4732, 
4777, 4813, 4870, 5149, 5163, 5272, 5273, 5282, 5283, 5357, 5358, 5396, 5512, 5524, 5601, 5636, 5653, 
5735, 5793, 5877, 5894, 5905, 5910, 5911, 5978, 6034, 6043, 6272, 6357, 6446, 6454, 6592, 6611, 6629, 
6656, 6715, 7232, 7235, 7392, 7541, 7726, 7937, 7938, 7995, 8007, 8073, 8117, 8308, 8415, 8423, 8428, 
8483, 8491, 8594, 8607, 8620, 8649, 8650, 8666, 8686, 8740, 9188, 9221, 9296, 9297, 9501, 9703, 9704, 
9715, 9858, 9964, 9968, 10057, 10061, 10070, 10094, 10116, 10124, 10136, 10236, 10471, 10478, 10479, 
10480, 10481, 10541, 10643, 10694, 10952, 10970, 11155, 11207, 11255, 11260, 11315]
Total extracted radiomics samples: 11190



Total samples with invalid mask: 95
Samples with invalid mask: [152, 158, 488, 1283, 1730, 1880, 1881, 2120, 2272, 2417, 2533, 2868, 3036, 
3335, 3526, 3825, 3996, 4030, 4207, 4213, 4483, 4544, 4653, 4732, 4777, 4813, 4870, 5149, 5163, 5283, 
5396, 5512, 5524, 5601, 5636, 5653, 5894, 5905, 5978, 6272, 6357, 6446, 6454, 6611, 6656, 6715, 7232, 
7235, 7392, 7541, 7726, 7937, 7995, 8007, 8117, 8308, 8423, 8428, 8483, 8491, 8594, 8607, 8620, 8649, 
8650, 8666, 8740, 9188, 9703, 9704, 9715, 9964, 9968, 10057, 10061, 10070, 10094, 10116, 10124, 10136, 
10236, 10471, 10478, 10479, 10481, 10541, 10643, 10694, 10952, 10970, 11155, 11207, 11255, 11260, 11315]
"""
