# -*- coding: utf-8 -*-
"""
Bar chart of U_m and U_v (Opstalt) using custom colours.
Saves figure as 'Um_Uv_opstalt_new.svg'.
"""

import numpy as np
import matplotlib.pyplot as plt

# ----- Your colours -----
TEAL_MED = "#5F938C"   # for U_v
ORANGE   = "#F79433"   # for U_m

# ----- Data -----
x_vals = np.array([40, 36, 32, 28, 24, 20, 16, 12, 8, 4])

Um = np.array([
    0.003555556,
    0.005472000,
    0.005916444,
    0.005400889,
    0.004352000,
    0.003111111,
    0.001934222,
    0.000992000,
    0.000369778,
    0.0000675556,
])

Uv = np.array([
    0.000258398,
    0.000465116,
    0.000620155,
    0.000723514,
    0.000775194,
    0.000775194,
    0.000723514,
    0.000620155,
    0.000465116,
    0.000258398,
])

# ----- Plot -----
fig, ax = plt.subplots(figsize=(8, 4))

width = 0.35  # bar width

ax.bar(x_vals - width/2, Um, width=width, color=ORANGE,   label="Bending Moment Deformation")
ax.bar(x_vals + width/2, Uv, width=width, color=TEAL_MED, label="Shear Deformation")

ax.set_title("Shear, and Bending Moment deformations, As a function of height")
ax.set_xlabel("H [m]")
ax.set_ylabel("U [m]")

# Ticks exactly at 40, 36, ..., 4
ax.set_xticks(x_vals)
ax.invert_xaxis()  # optional: keeps 40 on the left, 4 on the right

ax.legend()
ax.grid(axis="y", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig("Um_Uv_opstalt_new.svg", format="svg", bbox_inches="tight")
plt.show()
