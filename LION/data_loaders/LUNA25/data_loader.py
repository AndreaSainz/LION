# This file is part of LION library
# License : BSD-3
#
# Author  : Andrea Sainz
# Modifications: -
# =============================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
import warnings
from LION.utils.paths import LUNA25_PROCESSED_DATASET_PATH
from LION.utils.parameter import LIONParameter
import LION.CTtools.ct_geometry as ctgeo
import LION.CTtools.ct_utils as ct
from LION.CTtools.ct_utils import make_operator


class LUNA25(Dataset):
    def __init__(
        self,
        mode,
        geometry_parameters: ct.Geometry = None,
        parameters: LIONParameter = None,
    ):
        """
        Initializes LUNA25 dataset.

        Parameters:
            - device (torch.device): Selects the device to use for the data loader.
            - task (str): Defines pipeline on how to use data. Distinguish between "joint", "end_to_end", "segmentation", "reconstruction" and "diagnostic".
                          Dataset will return, for each task:
                          "segmentation"    -> (image, segmentation_label)
                          "reconstruction"  -> (sinogram, image_label)
                          "diagnostic"      -> (segmented_nodule, diagnostic_label)
                          "joint"           -> ?????
                          "end_to_end"      -> ?????
            - training_proportion (float): Defines training % of total data.
            - mode (str): Defines "train", "validation" or "test" mode.
            - Task (str): Defines what task is the Dataset being used for. "segmentation" (default) returns (gt_image,segmentation) pairs while "reconstruction" returns (sinogram, gt_image) pairs
            - annotation (str): Defines what annotation mode to use. Distinguish between "random" and "consensus". Default "consensus"
            - max_num_slices_per_patient (int): Defines the maximum number of slices to take per patient. Default is -1, which takes all slices we have of each patient and pcg_slices_nodule gets ignored.
            - pcg_slices_nodule (float): Defines percentage of slices with nodule in dataset. 0 meaning "no nodules at all" and 1 meaning "just take slices that contain annotated nodules". Only used if max_num_slices_per_patient != -1. Default is 0.5.
            - clevel (float): Defines consensus level if annotation=consensus. Value between 0-1. Default is 0.5.
            - geometry: Geometry() type, if sinograms are requied (e.g. fo "reconstruction")

        """
        # Input parsing
        assert mode in [
            "train",
            "validation",
            "test",
        ], f'Wrong mode argument, must be in ["train", "validation", "test"]'

        if parameters is None:
            parameters = LUNA25.default_parameters(geometry=geometry_parameters)
        self.params = parameters

        task = self.params.task
        assert task in [
            "joint",
            "end_to_end",
            "segmentation",
            "reconstruction",
            "diagnostic",
        ], f'task argument {task} not in ["joint", "end_to_end", "segmentation", "reconstruction", "diagnostic"]'

        if task not in ["segmentation", "reconstruction", "end_to_end"]:
            raise NotImplementedError(f"task {task} not implemented yet")

        if (
            task in ["reconstruction"]
            and geometry_parameters is None
            and self.params.geometry is None
        ):
            raise ValueError("geometry input required for recosntruction modes")

        # Aux variable setting
        self.sinogram_transform = None
        self.image_transform = None
        self.device = self.params.device

        if task in ["reconstruction"]:
            self.image_transform = ct.from_HU_to_mu
        if task in ["segmentation"]:
            self.image_transform = ct.from_HU_to_normal

        if geometry_parameters is not None:
            self.params.geometry = geometry_parameters
            self.operator = ct.make_operator(geometry_parameters)
        elif self.params.geometry is not None:
            self.operator = ct.make_operator(self.params.geometry)



    
    @staticmethod
    def default_parameters(geometry=None, task="reconstruction"):
        param = LIONParameter()
        param.name = "LUNA25 Data Loader"
        param.training_proportion = 0.8
        param.validation_proportion = 0.1
        param.testing_proportion = (
            1 - param.training_proportion - param.validation_proportion
        )  # not used, but for metadata
        param.max_num_slices_per_patient = 5
        param.pcg_slices_nodule = 0.5
        param.task = task
        param.folder = LUNA25_PROCESSED_DATASET_PATH
        if task == "reconstruction" and geometry is None:
            raise ValueError(
                "For reconstruction task geometry needs to be input to default_parameters(geometry=geometry_param)"
            )
        +





        self.pixels = 362               # Image resolution of 362x362 pixels on a domain size of 26x26 cm
        self.num_angles = 1000
        self.num_detectors = 513        # 513 equidistant detector bins s spanning the image diameter.
        self.src_orig_dist = 575
        self.src_det_dist = 1050
        self.slices_per_file = 128
        self.n_single_BP = int(n_single_BP)
        self.view_angles = int(view_angles)
        self.single_bp = single_bp
        self.sparse_view = sparse_view
        self.alpha = alpha