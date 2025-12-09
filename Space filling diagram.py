import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

# Settings
n_samples = 5000
dim = 2  # 2D parameter space

# --- Random sampling in [0,1]^2 ---
random_points = np.random.rand(n_samples, dim)

# --- Sobol sequence in [0,1]^2 ---
sobol_engine = qmc.Sobol(d=dim, scramble=True)
sobol_points = sobol_engine.random(n_samples)

# --- Plot: side-by-side 2D scatter plots ---
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Random sampling
axes[0].scatter(
    random_points[:, 0], random_points[:, 1],
    s=4, alpha=0.6,
    color="#B8D1CA"
)
axes[0].set_title("Random Sampling")
axes[0].set_xlabel("Parameter 1")
axes[0].set_ylabel("Parameter 2")
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)
axes[0].set_aspect("equal", adjustable="box")

# Sobol sequence
axes[1].scatter(
    sobol_points[:, 0], sobol_points[:, 1],
    s=4, alpha=0.6,
    color="#296872"
)
axes[1].set_title("Sobol Sequence")
axes[1].set_xlabel("Parameter 1")
axes[1].set_ylabel("Parameter 2")
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)
axes[1].set_aspect("equal", adjustable="box")

plt.tight_layout()

# Save as SVG
plt.savefig("random_vs_sobol_2D_colours.svg", format="svg")

# If you do not want to see the window, omit plt.show()
# plt.show()
