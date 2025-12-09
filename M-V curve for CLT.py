# -*- coding: utf-8 -*-
"""
3D surface plot with flipped axes and reversed custom colormap.
Deformations consistent with your C# Vandretdeformation.

Adds a colorbar as U-legend.
Saves figure as SVG.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import LinearSegmentedColormap


# ----------------------------------------------------
# REVERSED custom gradient colormap
# Highest U = darkest (#296872)
# Lowest U = lightest (#B8D1CA)
# ----------------------------------------------------
colors = ["#B8D1CA", "#5F938C", "#296872"]
custom_cmap = LinearSegmentedColormap.from_list("custom_hex_reversed", colors)


# ----------------------------------------------------
# Core functions (CONSISTENT WITH C#)
# ----------------------------------------------------
def Iz_from_AyNet(AyNet, L):
    AyNet_new = AyNet * 0.001
    return (AyNet_new / (L * 1000.0)) * (L ** 3)


def Uv(VEd, Hvaeg, G, b, L):
    return VEd * Hvaeg / (G * b * L * 1000.0)


def Um(MEd_y, Hvaeg, E0, Iz):
    return MEd_y * (Hvaeg ** 2) / (3.0 * E0 * Iz * 1000.0)


def Ut(ftd, Hvaeg, AntalSkruer_Lodret, L):
    if AntalSkruer_Lodret == 0:
        return 0.0
    return ftd * Hvaeg / (AntalSkruer_Lodret * 861.0 * 1.5 * L)


def Uk(VEd, Antalskruer_Vandret):
    if Antalskruer_Vandret == 0:
        return 0.0
    return VEd / (Antalskruer_Vandret * 375.0 * 1.5)


def U_total(VEd, MEd_y, ftd, Hvaeg, G, b, L, AyNet,
            AntalSkruer_Lodret, Antalskruer_Vandret, E0):
    Iz = Iz_from_AyNet(AyNet, L)
    return (
        Uv(VEd, Hvaeg, G, b, L)
        + Um(MEd_y, Hvaeg, E0, Iz)
        + Ut(ftd, Hvaeg, AntalSkruer_Lodret, L)
        + Uk(VEd, Antalskruer_Vandret)
    )


# ----------------------------------------------------
# 3D plot with custom reversed gradient
# ----------------------------------------------------
def plot_U_surface():
    # ---- Wall / material parameters ----
    Hvaeg = 12.0
    G = 800.0
    b = 0.12
    L = 3.0
    E0 = 11e3

    AyNet = 200 * L * 1000
    AntalSkruer_Lodret = 10
    Antalskruer_Vandret = 10

    # Axes ranges
    VEd_vals = np.linspace(0, 200, 150)
    MEd_vals = np.linspace(0, 3500, 150)

    # MEd = X, VEd = Y
    MEd_grid, VEd_grid = np.meshgrid(MEd_vals, VEd_vals)

    # tie force proportional to moment
    ftd_grid = 0.04 * MEd_grid

    U_grid = U_total(
        VEd_grid, MEd_grid, ftd_grid,
        Hvaeg, G, b, L, AyNet,
        AntalSkruer_Lodret, Antalskruer_Vandret, E0
    )

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        MEd_grid, VEd_grid, U_grid,
        cmap=custom_cmap, linewidth=0, antialiased=True
    )

    ax.set_xlabel("M", labelpad=10)
    ax.set_ylabel("V", labelpad=10)
    ax.set_zlabel("U total", labelpad=14)

    # ---- U legend / colorbar ----
    cbar = fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08)
    cbar.set_label("U total [-]", rotation=90, labelpad=12)

    # ---------------- Slight zoom-out ----------------
    x_min, x_max = 0.0, 3500.0
    y_min, y_max = 0.0, 200.0
    z_min, z_max = 0.0, float(np.max(U_grid))

    x_pad = 0.05 * (x_max - x_min)
    y_pad = 0.05 * (y_max - y_min)
    z_pad = 0.05 * (z_max - z_min)

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_zlim(z_min, z_max + z_pad)

    ax.dist = 11.5
    # -------------------------------------------------

    ax.view_init(elev=30, azim=225)

    fig.suptitle(
        "Displacements as a function of M and V - CLT Wall Assuming N = 0",
        y=0.95
    )

    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.02, top=0.90)

    # ---- SAVE AS SVG ----
    plt.savefig("U_surface_CLT_N0.svg", format="svg", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    plot_U_surface()

