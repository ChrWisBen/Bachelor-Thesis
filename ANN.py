"""
ANN surrogate model for a VERY slow 2D function + benchmarking + annotated plot.

- Trains an ANN surrogate for a complex function.
- Saves plots as an SVG figure.
- Benchmarks evaluation speed of:
    * complex_function_very_slow (artificially VERY expensive)
    * ANN surrogate (mlp.predict)
- Annotates the plot with timing and speed-up factor.

Run with:
    python ann_surrogate_benchmark_1000x_with_annotation.py

Dependencies:
    pip install numpy matplotlib scikit-learn
"""

import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error


# -------------------------------------------------------------------
# 1. Fast analytic function (ground truth for data)
# -------------------------------------------------------------------
def complex_function_fast(x, y):
    """
    Vectorized Franke-like function: smooth hills and valleys.
    Used as the mathematical 'truth'.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    term1 = 0.75 * np.exp(-(9 * x - 2) ** 2 / 4.0 - (9 * y - 2) ** 2 / 4.0)
    term2 = 0.75 * np.exp(-(9 * x + 1) ** 2 / 49.0 - (9 * y + 1) / 10.0)
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - (9 * y - 3) ** 2 / 4.0)
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


# -------------------------------------------------------------------
# 2. VERY slow version of the same function
# -------------------------------------------------------------------
def complex_function_very_slow(
    x,
    y,
    outer_loops: int = 70,
    inner_loops: int = 180,
):
    """
    Artificially *very* expensive 'true' function.

    Returns the SAME values as complex_function_fast, but uses:
      - Python loops over all points
      - Nested loops with heavy math (sin, cos, sqrt, pow)

    This mimics a very costly simulation you'd want to replace
    with a fast surrogate.

    You can crank outer_loops and inner_loops higher to make it
    arbitrarily slower (and thus increase speed-up from the ANN).
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    x_flat = x_arr.ravel()
    y_flat = y_arr.ravel()
    out = np.empty_like(x_flat, dtype=float)

    for i in range(x_flat.size):
        xi = float(x_flat[i])
        yi = float(y_flat[i])

        # Base value (same underlying function, scalar math)
        term1 = 0.75 * math.exp(-(9 * xi - 2) ** 2 / 4.0 - (9 * yi - 2) ** 2 / 4.0)
        term2 = 0.75 * math.exp(-(9 * xi + 1) ** 2 / 49.0 - (9 * yi + 1) / 10.0)
        term3 = 0.5 * math.exp(-(9 * xi - 7) ** 2 / 4.0 - (9 * yi - 3) ** 2 / 4.0)
        term4 = -0.2 * math.exp(-(9 * xi - 4) ** 2 - (9 * yi - 7) ** 2)
        base = term1 + term2 + term3 + term4

        # Massive dummy workload: outer * inner loops of transcendentals
        tmp = base
        for _ in range(outer_loops):
            for _ in range(inner_loops):
                tmp = math.sin(tmp) * math.cos(tmp) + math.sqrt(tmp * tmp + 1.0)
                tmp = tmp ** 0.5 + math.log(tmp * tmp + 1.0)

        # Keep original base as the "true" output
        out[i] = base

    return out.reshape(x_arr.shape)


# -------------------------------------------------------------------
# 3. Training / test data (from FAST function)
# -------------------------------------------------------------------
np.random.seed(42)

n_train = 4000
X_train = np.random.rand(n_train, 2)
y_train = complex_function_fast(X_train[:, 0], X_train[:, 1])

n_test = 2000
X_test = np.random.rand(n_test, 2)
y_test = complex_function_fast(X_test[:, 0], X_test[:, 1])


# -------------------------------------------------------------------
# 4. Build and train ANN surrogate
# -------------------------------------------------------------------
mlp = MLPRegressor(
    hidden_layer_sizes=(256, 256, 256),
    activation="relu",
    solver="adam",
    learning_rate="adaptive",
    learning_rate_init=1e-3,
    alpha=1e-6,
    batch_size=256,
    max_iter=4000,
    early_stopping=True,
    n_iter_no_change=80,
    validation_fraction=0.1,
    random_state=42,
    verbose=False,
)

mlp.fit(X_train, y_train)

# Evaluate surrogate quality
y_train_pred = mlp.predict(X_train)
y_test_pred = mlp.predict(X_test)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print("=== Surrogate accuracy ===")
print(f"Train R^2: {r2_train:.6f}")
print(f"Test  R^2: {r2_test:.6f}")
print(f"Train MSE: {mse_train:.6e}")
print(f"Test  MSE: {mse_test:.6e}")
print()


# -------------------------------------------------------------------
# 5. Benchmark: VERY slow function vs ANN surrogate
# -------------------------------------------------------------------
print("=== Benchmark: very slow true model vs ANN surrogate ===")

n_bench = 5000  # adjust as desired
X_bench = np.random.rand(n_bench, 2)
xb = X_bench[:, 0]
yb = X_bench[:, 1]

# Warm-up
_ = complex_function_very_slow(xb[:50], yb[:50])
_ = mlp.predict(X_bench[:50])

# Time very slow model
t0 = time.perf_counter()
_ = complex_function_very_slow(xb, yb)
t1 = time.perf_counter()
slow_time = t1 - t0

# Time surrogate
t0 = time.perf_counter()
_ = mlp.predict(X_bench)
t1 = time.perf_counter()
surrogate_time = t1 - t0

speedup = slow_time / surrogate_time if surrogate_time > 0 else float("inf")

print(f"Very slow true model time : {slow_time:.6f} s "
      f"for {n_bench} evaluations")
print(f"ANN surrogate time        : {surrogate_time:.6f} s "
      f"for {n_bench} evaluations")
print(f"Speed-up factor (slow / ANN): {speedup:.1f}x faster")
print()

# Text for annotating on the plot
speedup_text = (
    f"Benchmark ({n_bench} evals):\n"
    f"Slow model: {slow_time:.3f} s\n"
    f"Surrogate:  {surrogate_time:.3f} s\n"
    f"Speed-up ≈ {speedup:.0f}×"
)


# -------------------------------------------------------------------
# 6. Visualization grid (for plotting)
# -------------------------------------------------------------------
grid_size = 120
x = np.linspace(0, 1, grid_size)
y = np.linspace(0, 1, grid_size)
X_grid, Y_grid = np.meshgrid(x, y)

X_vis = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
y_vis_pred = mlp.predict(X_vis)

Z_true = complex_function_fast(X_grid, Y_grid)
Z_pred = y_vis_pred.reshape(grid_size, grid_size)
Z_error = np.abs(Z_pred - Z_true)


# -------------------------------------------------------------------
# 7. Custom colormap from your palette
# -------------------------------------------------------------------
palette_hex = ["#296872", "#5F938C", "#B8D1CA", "#F79433"]
surrogate_cmap = LinearSegmentedColormap.from_list("surrogate_cmap", palette_hex)


# -------------------------------------------------------------------
# 8. Plotting + SVG export (with speedup annotation)
# -------------------------------------------------------------------
plt.style.use("default")
fig = plt.figure(figsize=(18, 6))

# --- Plot 1: True function + training points ---
ax1 = fig.add_subplot(1, 3, 1, projection="3d")
ax1.plot_surface(
    X_grid,
    Y_grid,
    Z_true,
    cmap=surrogate_cmap,
    linewidth=0,
    antialiased=True,
    alpha=0.9,
)
ax1.scatter(
    X_train[:, 0],
    X_train[:, 1],
    y_train,
    color="#F79433",
    s=8,
    alpha=0.8,
    label="Training samples",
)

ax1.set_title("True Complex Function", fontsize=13)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("f(x, y)")
ax1.view_init(elev=35, azim=-60)
ax1.legend(loc="upper right", fontsize=8)

# --- Plot 2: ANN surrogate + accuracy & speedup ---
ax2 = fig.add_subplot(1, 3, 2, projection="3d")
ax2.plot_surface(
    X_grid,
    Y_grid,
    Z_pred,
    cmap=surrogate_cmap,
    linewidth=0,
    antialiased=True,
    alpha=0.95,
)

ax2.set_title("ANN Surrogate Prediction", fontsize=13)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("ŷ(x, y)")
ax2.view_init(elev=35, azim=-60)

# Annotate both accuracy and performance in the 2D overlay
ax2.text2D(
    0.05,
    0.95,
    f"Train R² = {r2_train:.4f}\nTest  R² = {r2_test:.4f}",
    transform=ax2.transAxes,
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

ax2.text2D(
    0.05,
    0.70,
    speedup_text,
    transform=ax2.transAxes,
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

# --- Plot 3: Absolute error heatmap ---
ax3 = fig.add_subplot(1, 3, 3)
contour = ax3.contourf(
    X_grid,
    Y_grid,
    Z_error,
    levels=40,
    cmap=surrogate_cmap,
)
cbar = fig.colorbar(contour, ax=ax3)
cbar.set_label("|f(x, y) - ŷ(x, y)|")

ax3.set_title("Surrogate Error (ANN vs True)", fontsize=13)
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_aspect("equal")

plt.tight_layout()
plt.savefig("ann_surrogate_very_slow_annotated.svg", format="svg")
# plt.show()  # optional
