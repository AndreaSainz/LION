from radiomics import firstorder, shape, shape2D, glcm, glszm, glrlm, ngtdm, gldm
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

save_fig_path = "scripts/pipeline_project2"

# Define experiment
experiment = ct_experiments.NormalDoseCTSegmentation(dataset="LIDC-IDRI")

# %% Dataset
lidc_dataset = experiment.get_training_dataset()

#%% Define DataLoader
batch_size = 1
lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=False)
lidc_test = DataLoader(experiment.get_testing_dataset(), batch_size, True)
lidc_val = DataLoader(experiment.get_validation_dataset(), batch_size, True)


# TRAIN - Corrupt sample 131: [Errno 2] No such file or directory: '/store/LION/datasets/processed/LIDC-IDRI/LIDC-IDRI-0027/mask_75_nodule_0_annotation_306.npy'
# TRAIN - Corrupt sample 133: [Errno 2] No such file or directory: '/store/LION/datasets/processed/LIDC-IDRI/LIDC-IDRI-0027/mask_92_nodule_3_annotation_297.npy'
# TRAIN bad indices: [131, 133]
# TEST bad indices: []
# VALIDATION bad indices: []


for batch_id, batch in enumerate(lidc_dataloader):

    if batch_id % 20 == 0:
        sinogram = batch[0]
        mask_tensor = batch[1]
        reconstruction_tensor = batch[2]
        diagnosis = batch[3]

        # shapes of tensors
        print(f" {batch_id} Sinogram shape:", sinogram.shape)
        print(f" {batch_id} Mask shape:", mask_tensor.shape)
        print(f" {batch_id} Image shape:", reconstruction_tensor.shape)
        print(f" {batch_id} Diagnosis:", diagnosis.shape)

        # check that all slcies have tumors
        tumor_pixels = mask_tensor[0, 1].sum()

        if tumor_pixels == 0:
            print(f" {batch_id} Found slice without tumour:")

        # visualize sinogram
        # plt.figure(figsize=(10,4))
        # plt.imshow(
        #     sinogram[0].squeeze().cpu().numpy(),
        #     cmap="gray",
        #     aspect="auto"
        # )
        # plt.title("Sinogram")
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig(f"{save_fig_path}/sinogram_{batch_id}.png", dpi=300)
        # plt.close()

        # visualice image and image+mask
        image_plot = reconstruction_tensor[0, 0].cpu().numpy()

        mask_plot = mask_tensor[0, 1].cpu().numpy()

        plt.figure(figsize=(16, 5))

        plt.subplot(1, 4, 1)
        plt.imshow(image_plot, cmap="gray")
        plt.title("CT slice")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.imshow(mask_plot, cmap="gray")
        plt.title("Tumour mask")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.imshow(image_plot, cmap="gray")
        plt.imshow(mask_plot, alpha=0.4)
        plt.title("Overlay")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.imshow(sinogram[0].squeeze().cpu().numpy(), cmap="gray", aspect="auto")
        plt.title("Sinogram")
        plt.colorbar()
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"{save_fig_path}/image_and_mask_{batch_id}.png", dpi=300)
        plt.close()
