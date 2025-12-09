# -*- coding: utf-8 -*-
"""
Bar graph (histogram) from Excel:
- y-axis: number of observations
- x-axis: value bins grouped by 10s
Saves as SVG too.

Edit:
    excel_path
    sheet_name (if needed)
    value_column (name or index)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------
# Settings (EDIT THESE)
# ---------------------------
excel_path = Path(r"C:\Users\ChristianWBendtsen\OneDrive - ABC\Skrivebord\Bachelor\RC_Wall_Cap_Samples\Forskydning_kN.xlsx")  # <-- CHANGE
sheet_name = 0                                  # or "Sheet1"
value_column = 0                                # column NAME or index

bin_width = 250                               # group by 10s
save_svg = True
out_svg = "Maximum_Additional_Shear.svg"

# Your colors (optional but consistent)
TEAL_DARK = "#296872"
MINT_LIGHT = "#B8D1CA"

# ---------------------------
# Load Excel
# ---------------------------
df = pd.read_excel(excel_path, sheet_name=sheet_name)

# Pick column (works for both name and index)
if isinstance(value_column, int):
    vals = df.iloc[:, value_column]
else:
    vals = df[value_column]

# Convert comma decimals + force numeric
vals = (vals.astype(str)
             .str.replace(",", ".", regex=False))
vals = pd.to_numeric(vals, errors="coerce").dropna()

if len(vals) == 0:
    raise ValueError("No numeric values found in the selected column.")

# ---------------------------
# Bin by 10s
# ---------------------------
vmin = np.floor(vals.min() / bin_width) * bin_width
vmax = np.ceil(vals.max() / bin_width) * bin_width
bins = np.arange(vmin, vmax + bin_width, bin_width)

counts, edges = np.histogram(vals, bins=bins)
centers = (edges[:-1] + edges[1:]) / 2

# Labels like "0–10", "10–20", ...
labels = [f"{int(edges[i])}–{int(edges[i+1])}" for i in range(len(edges)-1)]

# ---------------------------
# Plot
# ---------------------------
fig, ax = plt.subplots(figsize=(9, 4))

ax.bar(centers, counts, width=bin_width*0.9,
       color=MINT_LIGHT, edgecolor=TEAL_DARK, linewidth=1.5)

ax.set_xlabel(f"Value [kN] (grouped by {bin_width})")
ax.set_ylabel("Number of observations")
ax.set_title("Bar Chart - RC Wall Shear Capacity")

ax.set_xticks(centers)
ax.set_xticklabels(labels, rotation=45, ha="right")

ax.grid(axis="y", linestyle=":", alpha=0.5)
plt.tight_layout()

if save_svg:
    plt.savefig(out_svg, format="svg", bbox_inches="tight")
    print(f"Saved SVG: {out_svg}")

plt.show()
