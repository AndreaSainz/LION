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
from LION.utils.paths import DETECT_PROCESSED_DATASET_PATH
from LION.utils.parameter import LIONParameter
import LION.CTtools.ct_geometry as ctgeo
import LION.CTtools.ct_utils as ct
from LION.CTtools.ct_utils import make_operator


class deteCT(Dataset):
    def __init__(
        self,
        mode,
        geometry_params: ct.Geometry = None,
        parameters: LIONParameter = None,
    ):