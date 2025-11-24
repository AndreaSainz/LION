import numpy as np
import pandas as pd

meta = np.load("/store/LION/as3628/datasets/LUNA25/raw/luna25_nodule_blocks/metadata/122923_1_20010102.npy", allow_pickle=True).item()

df = pd.DataFrame([meta])
df.to_csv("ejemplo_csv.csv", index=False)