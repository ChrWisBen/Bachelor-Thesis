# -*- coding: utf-8 -*-
"""
PCA pipeline (Excel) in your style + plots in your colors.

Features:
- Reads Excel without trusting headers
- If first row looks like variable names, it becomes header and is removed from data
- Converts comma decimals to floats
- Drops all-NaN + constant columns
- Optional IQR outlier removal
- Correlation heatmap:
    * REVERSED colors NOW:
        +1 correlation = ORANGE
        -1 correlation = TEAL
        0 correlation  = WHITE
    * Nonlinear SymLog scaling to make low values more visible
    * Colorbar ticks in decimals + more tick points (denser near 0)
    * Removes stray "0" axis tick by forcing tick labels to feature names only
    * SAVED AS SVG with current name + _CLT
- PCA cumulative explained variance plot (TEAL) SAVED AS SVG with current name + _CLT
- PC1 vs PC2 scatter SAVED AS SVG with current name + _CLT
- Loadings table + optional export
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from pathlib import Path


# ---------------------------
# Helpers
# ---------------------------
def remove_outliers_iqr(df, columns, k=1.5):
    filtered_df = df.copy()
    for col in columns:
        data = pd.to_numeric(filtered_df[col], errors='coerce')
        if data.notna().sum() < 2:
            continue
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        filtered_df = filtered_df[(data >= lower) & (data <= upper)]
    return filtered_df


def convert_comma_decimals(df):
    df = df.copy()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = (
            df[col].astype(str)
                  .str.replace(',', '.', regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def usable_numeric_columns(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    usable_cols = [
        c for c in numeric_cols
        if df[c].notna().sum() >= 2 and df[c].nunique(dropna=True) > 1
    ]
    return usable_cols


def first_row_is_header(df):
    r0 = df.iloc[0].astype(str)
    return r0.str.contains(r"[A-Za-z]", regex=True).any()


# ---------------------------
# Settings
# ---------------------------
excel_path = Path(r"C:\Users\ChristianWBendtsen\OneDrive - ABC\Skrivebord\Bachelor\CLT_Wall_Cap_Samples\CLT_Wall_Samples_Copy.xlsx")
sheet_name = 0

showplots = True
doexport = True
remove_outliers = False
iqr_k = 1.5

save_svgs = True
out_dir = Path(".")  # folder for svg outputs (change if you want)

# ---- SVG names: current name + "_CLT.svg" ----
def svg_name(stem):
    return out_dir / f"{stem}_CLT.svg"

heatmap_svg  = svg_name("corr_heatmap")
variance_svg = svg_name("pca_variance_retention")
pc12_svg     = svg_name("pc1_vs_pc2")


print("Exists?", excel_path.exists())
if not excel_path.exists():
    raise FileNotFoundError(excel_path)


# ---------------------------
# 1) Load Excel WITHOUT trusting headers
# ---------------------------
df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
df = df.dropna(axis=1, how="all")

if first_row_is_header(df):
    df.columns = df.iloc[0].astype(str).str.strip()
    df = df.iloc[1:].reset_index(drop=True)
else:
    df.columns = [f"x{i+1}" for i in range(df.shape[1])]

df.columns = df.columns.astype(str)

print("\nColumns:", df.columns.tolist())
print(df.head())


# ---------------------------
# 2) Convert comma decimals -> floats
# ---------------------------
df = convert_comma_decimals(df)
original_df = df.copy()


# ---------------------------
# 3) Pick usable numeric columns
# ---------------------------
usable_cols = usable_numeric_columns(original_df)
print("\nUsable numeric columns:", usable_cols)

if len(usable_cols) == 0:
    raise ValueError("No usable numeric columns found. Check your sheet.")


# ---------------------------
# 4) Optional outlier removal
# ---------------------------
if remove_outliers:
    before = len(original_df)
    original_df = remove_outliers_iqr(original_df, usable_cols, k=iqr_k)
    after = len(original_df)
    print(f"\nOutlier removal: {before} -> {after} rows")


# ---------------------------
# 5) Correlation heatmap (YOUR COLORS, NOW REVERSED) + boosted low values
#    +1 ORANGE, -1 TEAL, 0 WHITE
#    SAVED AS SVG
#    Removes stray "0" tick by forcing tick labels to feature names only
# ---------------------------
if showplots:
    TEAL_DARK = "#296872"
    ORANGE    = "#F79433"

    # ---- REVERSED ORDER HERE (TEAL -> WHITE -> ORANGE) ----
    # Means: -1 teal, 0 white, +1 orange
    custom_heatmap_cmap = LinearSegmentedColormap.from_list(
        "teal_white_orange",
        [TEAL_DARK, "#FFFFFF", ORANGE],
        N=256
    )

    corr = original_df[usable_cols].corr()

    norm = mcolors.SymLogNorm(
        linthresh=0.05,
        linscale=1.0,
        vmin=-1, vmax=1,
        base=10
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    hm = sns.heatmap(
        corr,
        cmap=custom_heatmap_cmap,
        norm=norm,
        annot=False,
        square=True,
        cbar_kws={"label": "Correlation (−1 teal to +1 orange), SymLog scaled"},
        ax=ax
    )

    # ---- Force ticks to be ONLY feature names (removes stray "0") ----
    cols = corr.columns.tolist()
    rows = corr.index.tolist()

    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_yticks(np.arange(len(rows)) + 0.5)

    ax.set_xticklabels(cols, rotation=90, ha="center", va="top")
    ax.set_yticklabels(rows, rotation=0)

    # ---- More colorbar ticks + decimal formatting ----
    cbar = hm.collections[0].colorbar
    ticks = [-1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])

    ax.set_title("Correlation Heatmap of Input Features - CLT Wall Capacity", fontsize=14)
    fig.tight_layout()

    if save_svgs:
        fig.savefig(heatmap_svg, format="svg", bbox_inches="tight")
        print(f"Saved heatmap SVG: {heatmap_svg}")

    plt.show()


# ---------------------------
# 6) PCA (drop NaN rows first)
# ---------------------------
X = original_df[usable_cols].copy()

before_rows = len(X)
X = X.dropna(axis=0, how="any")
after_rows = len(X)
if after_rows < before_rows:
    print(f"\nDropped NaN rows before PCA: {before_rows} -> {after_rows}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_
cum_explained = np.cumsum(explained)

print("\nExplained variance ratio per PC:")
for i, v in enumerate(explained, start=1):
    print(f"  PC{i}: {v:.4f}")

print("\nCumulative explained variance:")
for i, v in enumerate(cum_explained, start=1):
    print(f"  PC1..PC{i}: {v:.4f}")


# ---------------------------
# 7) PCA variance plot (YOUR TEAL COLORS) SAVED AS SVG
# ---------------------------
if showplots:
    TEAL_DARK  = "#296872"
    TEAL_MED   = "#5F938C"
    MINT_LIGHT = "#B8D1CA"

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(
        range(1, len(cum_explained) + 1),
        cum_explained,
        marker='o',
        color=TEAL_DARK,
        markerfacecolor=TEAL_MED,
        markeredgecolor=TEAL_DARK,
        linewidth=2
    )
    ax2.fill_between(
        range(1, len(cum_explained) + 1),
        cum_explained,
        color=MINT_LIGHT,
        alpha=0.4
    )

    ax2.set_xlabel('Number of Principal Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('PCA Variance Retention - CLT Wall Capacity')
    ax2.set_ylim(0, 1.02)
    ax2.grid(True, linestyle=":")
    fig2.tight_layout()

    if save_svgs:
        fig2.savefig(variance_svg, format="svg", bbox_inches="tight")
        print(f"Saved variance SVG: {variance_svg}")

    plt.show()


# ---------------------------
# 8) PC1 vs PC2 scatter SAVED AS SVG
# ---------------------------
if showplots:
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    ax3.scatter(X_pca[:, 0], X_pca[:, 1], s=35)
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_title("PCA: PC1 vs PC2")
    ax3.grid(True, linestyle=":")
    fig3.tight_layout()

    if save_svgs:
        fig3.savefig(pc12_svg, format="svg", bbox_inches="tight")
        print(f"Saved PC1-vs-PC2 SVG: {pc12_svg}")

    plt.show()


# ---------------------------
# 9) Loadings
# ---------------------------
loadings = pd.DataFrame(
    pca.components_.T,
    index=usable_cols,
    columns=[f"PC{i}" for i in range(1, len(usable_cols) + 1)]
)

print("\nLoadings (first 5 PCs):")
print(loadings.iloc[:, :5])


# ---------------------------
# 10) Optional export
# ---------------------------
if doexport:
    scores = pd.DataFrame(X_pca, columns=[f"PC{i}" for i in range(1, len(usable_cols) + 1)])
    loadings.to_excel(out_dir / "PCA_loadings.xlsx")
    scores.to_excel(out_dir / "PCA_scores.xlsx")
    print("\nSaved: PCA_loadings.xlsx and PCA_scores.xlsx")

