import pandas as pd
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


csv_path = "/store/LION/as3628/nosnap/LUNA25/preprocessed/LUNA25_master.csv"

df = pd.read_csv(csv_path)

# delete sames CT files and extract from them el path, the dimensions of the CT and the dimensions of the voxels
unique_cts = df.drop_duplicates(subset="ct_npy_path")[["ct_npy_path", "ct_dims", "ct_spacing"]]

# Convert text tu tuples
unique_cts["ct_dims"] = unique_cts["ct_dims"].apply(eval)
unique_cts["ct_spacing"] = unique_cts["ct_spacing"].apply(eval)



log_path = "resumen_cts.log"
with open(log_path, "w") as f:
    f.write(unique_cts.head().to_string() + "\n")
    f.write(f"Total CTs: {len(unique_cts)}\n\n")

    f.write("=== SHAPES ===\n")
    # Counting how many different shapes we have
    shape_counts = Counter(unique_cts["ct_dims"])
    for s, c in shape_counts.items():
        f.write(f"{s}: {c} CTs\n")

    f.write("\n=== SPACING ===\n")
    # counting how many different sizes we hace 
    spacing_counts = Counter(unique_cts["ct_spacing"])

    #substract sx,sy,sz for histograms
    spacings = np.array(unique_cts["ct_spacing"].tolist())
    sx, sy, sz = spacings[:,0], spacings[:,1], spacings[:,2]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.hist(sx, bins=50)
    plt.title("sx")
    plt.xlabel("mm")
    plt.ylabel("Número de CTs")

    plt.subplot(1, 3, 2)
    plt.hist(sy, bins=50)
    plt.title("sy")
    plt.xlabel("mm")

    plt.subplot(1, 3, 3)
    plt.hist(sz, bins=50)
    plt.title("sz")
    plt.xlabel("mm")

    plt.tight_layout()
    plt.savefig("spacing_hist.png", dpi=200)

    # printing diferent spacing with the number of CTs

    for s, c in spacing_counts.items():
        f.write(f"{s}: {c} CTs\n")