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
from LION.classical_algorithms.fdk import fdk
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

# Define your data paths
savefolder = pathlib.Path("/store/LION/as3628/trained_models/")


def my_ssim(x, y):
    x = x.cpu().numpy().squeeze()
    y = y.cpu().numpy().squeeze()
    return ssim(x, y, data_range=x.max() - x.min())


save_fig_path = "scripts/pipeline_project2"

# Define experiment
experiment = ct_experiments.NormalDoseCTSegmentation(dataset="LIDC-IDRI")

# extract the operator for the experiment
operator = experiment.geometry

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
    if batch_id < 200:
        sinogram = batch[0]
        mask_tensor = batch[1]
        reconstruction_tensor = batch[2]
        diagnosis = batch[3]

        image_np = reconstruction_tensor[0, 0].cpu().numpy()
        mask_np = mask_tensor[0, 1].cpu().numpy().astype("uint8")

        # Check if mask has the expected dimensionality and shape or if there is a nodule
        if mask_np.ndim != 2 or mask_np.shape != image_np.shape or mask_np.sum() == 0:
            continue

        # checking the mask is not just a line (all pixels in one row/column)
        rows, cols = np.where(mask_np > 0)

        is_horizontal_line = len(np.unique(rows)) == 1
        is_vertical_line = len(np.unique(cols)) == 1

        if is_horizontal_line or is_vertical_line:
            continue

        image_sitk = sitk.GetImageFromArray(image_np)
        mask_sitk = sitk.GetImageFromArray(mask_np)

        assert mask_np.ndim == 2
        assert mask_np.shape == image_np.shape
        assert mask_sitk.GetDimension() == 2

        # reconstructing image from sinogram
        reconstruction = fdk(sinogram, operator)

        # changing the recontruction type to stick
        reconstruction_np = reconstruction[0, 0].detach().cpu().numpy()
        reconstruction_sitk = sitk.GetImageFromArray(reconstruction_np)

        print("MASK")
        print("  shape:", mask_np.shape)
        print("  unique:", np.unique(mask_np))
        print("  nonzero:", np.count_nonzero(mask_np))

        print("RECONSTRUCTION")
        print("  shape:", reconstruction_np.shape)
        print("  min:", np.nanmin(reconstruction_np))
        print("  max:", np.nanmax(reconstruction_np))
        print("  NaN:", np.isnan(reconstruction_np).sum())
        print("  Inf:", np.isinf(reconstruction_np).sum())

        result = extractor.execute(reconstruction_sitk, mask_sitk)

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
            kde=False,
            stat="density",
            common_norm=False,
        )

        plt.title(f"{feature}: benign vs malignant")
        plt.xlabel(feature)
        plt.ylabel("Density")

        plt.tight_layout()
        plt.savefig(f"{save_fig_path}_bvsm_{feature}.png", dpi=300)
        plt.close()
