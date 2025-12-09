import numpy as np
import matplotlib.pyplot as plt

# Create figure and axis
fig, ax = plt.subplots(figsize=(5, 4))

ax.set_xlabel("Model complexity")
ax.set_ylabel("Error")
ax.set_title("Training vs validation error\nas a function of model complexity", fontsize=9)

# Schematic complexity axis
x = np.linspace(0, 1, 200)

# Schematic training error: monotonically decreasing
train_err = 0.25 - 0.20 * x + 0.05 * x**2

# Schematic validation error: U-shaped
val_err = 0.18 - 0.15 * x + 0.30 * (x - 0.5)**2

# Plot curves with your colours
ax.plot(x, train_err, label="Training error", color="#B8D1CA")
ax.plot(x, val_err, label="Validation error", color="#296872")

# Mark approximate optimal complexity (minimum of validation error)
opt_index = np.argmin(val_err)
opt_x = x[opt_index]
opt_y = val_err[opt_index]

ax.axvline(opt_x, linestyle="--", linewidth=1.0, color="#296872")
ax.scatter([opt_x], [opt_y], color="#296872", s=25)

ax.text(
    opt_x, opt_y + 0.02,
    "Selected model\n(best validation error)",
    fontsize=7, ha="center", va="bottom"
)

# Annotate underfitting and overfitting regions
ax.text(
    0.05, val_err[0] + 0.03,
    "Underfitting\n(high bias)",
    fontsize=7, ha="left", va="bottom"
)
ax.text(
    0.80, val_err[-1] + 0.03,
    "Overfitting\n(high variance)",
    fontsize=7, ha="left", va="bottom"
)

ax.legend(fontsize=7)
ax.set_xlim(0, 1)

plt.tight_layout()

# Show or save
plt.show()
# For thesis use, you may want:
# plt.savefig("generalisation_error_curve.svg", format="svg", bbox_inches="tight")
# plt.savefig("generalisation_error_curve.png", dpi=300, bbox_inches="tight")

