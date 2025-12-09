# -*- coding: utf-8 -*-
"""
Grouped bar chart with two series only, using your colours.
Series values updated per your list.

Saves figure as 'two_bars_opstalt.svg'.
"""

import numpy as np
import matplotlib.pyplot as plt

# ----- Your colours -----
TEAL_MED = "#5F938C"   # Series 2
ORANGE   = "#F79433"   # Series 1

# ----- X values -----
x_vals = np.array([40, 36, 32, 28, 24, 20, 16, 12, 8, 4])

# ----- Two bar series (updated) -----
series1 = np.array([  # first column
    0.002000000,
    0.002952000,
    0.002929778,
    0.002352000,
    0.001552000,
    0.000777778,
    0.000192000,
    0.0,
    0.0,
    0.0,
])

series2 = np.array([  # second column
    0.000129199,
    0.000232558,
    0.000310078,
    0.000361757,
    0.000387597,
    0.000387597,
    0.000361757,
    0.000310078,
    0.000232558,
    0.000129199,
])

# ----- Plot -----
fig, ax = plt.subplots(figsize=(8, 4))

width = 0.35  # bar width

ax.bar(x_vals - width/2, series1, width=width,
       color=ORANGE, label="Effective Moment Deformations")
ax.bar(x_vals + width/2, series2, width=width,
       color=TEAL_MED, label="Shear Deformations")

ax.set_title("Effective Bending Moment Deformations, and Shear Deformations")
ax.set_xlabel("H [m]")
ax.set_ylabel("U [m]")

# Ticks exactly at 40, 36, ..., 4
ax.set_xticks(x_vals)
ax.invert_xaxis()  # 40 on the left, 4 on the right

ax.legend()
ax.grid(axis="y", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig("two_bars_opstalt.svg", format="svg", bbox_inches="tight")
plt.show()

