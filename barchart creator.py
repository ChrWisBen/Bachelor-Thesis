# -*- coding: utf-8 -*-
"""
Bar graph (histogram) from Excel:
- y-axis: number of observations
- x-axis: value bins grouped by bin_width
- Lets you choose which column to plot (by name or index)
- Saves as SVG with a name you can change in ONE place.

Edit:
    excel_path
    sheet_name
    value_column   <-- set to column NAME (string) or index (int)
    bin_width
    FIG_NAME_STEM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------
# Settings (EDIT THESE)
# ---------------------------
excel_path = Path(
    r"C:\Users\ChristianWBendtsen\OneDrive - ABC\Skrivebord\Bachelor\RC_Wall_Cap_Samples\KunNorm_RC.xlsx"
)
sheet_name = 0

# ---- Choose column here ----
# Option A: by column NAME, e.g. "Utilization"
# value_column = "Utilization"
# Option B: by column INDEX, e.g. 0 for first column
value_column = 16
# ---------------------------

bin_width = 0.1     # e.g. 0.01 for utilization, 10 / 50 / 250 for forces etc.
save_svg = True

# ---- OUTPUT NAME (edit only this) ----
FIG_NAME_STEM = "Utilization_Histogram_VEd_Additional "   # no extension
FIG_SUFFIX = "RC"                          # e.g. "_RC" or "_CLT"
out_svg = f"{FIG_NAME_STEM}{FIG_SUFFIX}.svg"
# --------------------------------------

# Your colors
TEAL_DARK  = "#296872"
MINT_LIGHT = "#B8D1CA"

# ---------------------------
# Load Excel
# ---------------------------
df = pd.read_excel(excel_path, sheet_name=sheet_name)

print("\nAvailable columns in Excel:")
for i, c in enumerate(df.columns):
    print(f"  {i}: {c}")

def resolve_column(df, col_spec):
    """
    col_spec can be:
      - int: column index
      - str: column name
      - None: auto-pick first numeric column
    """
    if col_spec is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            raise ValueError("No numeric columns found. Set value_column manually.")
        return num_cols[0]

    if isinstance(col_spec, int):
        if col_spec < 0 or col_spec >= df.shape[1]:
            raise IndexError(f"value_column index {col_spec} out of range.")
        return df.columns[col_spec]

    if isinstance(col_spec, str):
        if col_spec not in df.columns:
            raise KeyError(f"Column '{col_spec}' not found. See printed list above.")
        return col_spec

    raise TypeError("value_column must be int, str, or None.")

col_name = resolve_column(df, value_column)
vals = df[col_name]

# Convert comma decimals + force numeric
vals = (
    vals.astype(str)
        .str.replace(",", ".", regex=False)
)
vals = pd.to_numeric(vals, errors="coerce").dropna()

if len(vals) == 0:
    raise ValueError("No numeric values found in the selected column.")

# ---------------------------
# Bin values
# ---------------------------
vmin = np.floor(vals.min() / bin_width) * bin_width
vmax = np.ceil(vals.max() / bin_width) * bin_width
bins = np.arange(vmin, vmax + bin_width, bin_width)

counts, edges = np.histogram(vals, bins=bins)
centers = (edges[:-1] + edges[1:]) / 2

# Nice float labels for small bin widths
def fmt(x):
    # switch formatting depending on scale
    return f"{x:.3f}" if bin_width < 1 else f"{x:.0f}"

labels = [f"{fmt(edges[i])}–{fmt(edges[i+1])}" for i in range(len(edges) - 1)]

# ---------------------------
# Plot
# ---------------------------
fig, ax = plt.subplots(figsize=(9, 4))

ax.bar(
    centers, counts,
    width=bin_width * 0.9,
    color=MINT_LIGHT,
    edgecolor=TEAL_DARK,
    linewidth=1.5
)

ax.set_xlabel(f"{col_name} (grouped by {bin_width})")
ax.set_ylabel("Number of observations")
ax.set_title(f"Histogram of {col_name} - RC")

ax.set_xticks(centers)
ax.set_xticklabels(labels, rotation=45, ha="right")

ax.grid(axis="y", linestyle=":", alpha=0.5)
plt.tight_layout()

if save_svg:
    plt.savefig(out_svg, format="svg", bbox_inches="tight")
    print(f"Saved SVG: {out_svg}")

plt.show()
