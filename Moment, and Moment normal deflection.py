# -*- coding: utf-8 -*-
"""
Bar chart of bending moment deflection and effective bending moment deflection
using custom colours.

Data updated with:
    H, U_m, U_eff_raw, U_eff_clipped

Saves figure as 'Um_Uv_opstalt_final.svg'.
"""

import numpy as np
import matplotlib.pyplot as plt

# ----- Your colours -----
TEAL_MED = "#5F938C"   # for effective bending moment deflection
ORANGE   = "#F79433"   # for bending moment deflection

# ----- Data -----
x_vals = np.array([40, 36, 32, 28, 24, 20, 16, 12, 8, 4])

# Bending moment deflection (U_m)
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

# Effective bending moment deflection (raw, can be negative)
Ueff_raw = np.array([
    0.002000000,
    0.002952000,
    0.002929778,
    0.002352000,
    0.001552000,
    0.000777778,
    0.000192000,
   -0.000128000,
   -0.000190222,
   -0.000088000,
])

# Effective bending moment deflection (clipped at 0, for plotting)
Ueff = np.array([
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

# ----- Plot -----
fig, ax = plt.subplots(figsize=(8, 4))

width = 0.35  # bar width

ax.bar(x_vals - width/2, Um,   width=width,
       color=ORANGE,   label="Bending moment deflection")
ax.bar(x_vals + width/2, Ueff, width=width,
       color=TEAL_MED, label="Effective bending moment deflection")

ax.set_title("Bending moment and effective bending moment deflection")
ax.set_xlabel("H [m]")
ax.set_ylabel("U [m]")

# Ticks exactly at 40, 36, ..., 4
ax.set_xticks(x_vals)
ax.invert_xaxis()  # 40 on the left, 4 on the right

ax.legend()
ax.grid(axis="y", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig("Um_Uv_opstalt_final.svg", format="svg", bbox_inches="tight")
plt.show()

