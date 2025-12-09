import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------- Colours ----------
TEAL_DARK  = "#296872"
TEAL_MED   = "#5F938C"
MINT_LIGHT = "#B8D1CA"
ORANGE     = "#F79433"

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")

def add_box(x, y, w, h, color, text, fontsize=10):
    rect = Rectangle((x, y), w, h,
                     linewidth=1.5,
                     edgecolor="black",
                     facecolor=color)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text,
            ha="center", va="center", fontsize=fontsize)

def add_arrow(p_from, p_to, text=None, fontsize=9, dy_text=0.2):
    ax.annotate(
        "",
        xy=p_to,
        xytext=p_from,
        arrowprops=dict(arrowstyle="->", linewidth=1.5, color="black")
    )
    if text is not None:
        x_mid = 0.5 * (p_from[0] + p_to[0])
        y_mid = 0.5 * (p_from[1] + p_to[1])
        ax.text(x_mid, y_mid + dy_text, text,
                ha="center", va="center", fontsize=fontsize)

# ---------------- Top row: offline training ----------------
y_top = 3.0
h = 1.0
w = 1.8

# 1) Input samples
add_box(0.5, y_top, w, h, MINT_LIGHT,
        "Sampled inputs\nxᵢ")

# 2) Expensive model
add_box(3.0, y_top, w, h, ORANGE,
        "Expensive model\nFEM / CFD /\nnonlinear")

# 3) Training data
add_box(5.5, y_top, w, h, TEAL_MED,
        "Training data\n(xᵢ, yᵢ)")

# 4) Surrogate training
add_box(8.0, y_top, w, h, TEAL_DARK,
        "Surrogate-assisted\nmachine learning\n(training)")

# Arrows (top row)
add_arrow((0.5 + w, y_top + h/2),
          (3.0,      y_top + h/2),
          "yᵢ = f(xᵢ)")
add_arrow((3.0 + w, y_top + h/2),
          (5.5,      y_top + h/2))
add_arrow((5.5 + w, y_top + h/2),
          (8.0,      y_top + h/2),
          "{(xᵢ, yᵢ)}")

ax.text(3.5, 4.4,
        "Offline: generate data with expensive model and train surrogate",
        ha="center", va="center", fontsize=9)

# ---------------- Bottom row: online prediction ----------------
y_bot = 1.0

# 5) New input
add_box(2.0, y_bot, w, h, MINT_LIGHT,
        "New input\nx*")

# 6) Trained surrogate
add_box(5.0, y_bot, w, h, TEAL_DARK,
        "Trained surrogate\nŷ = f̂(x)")

# 7) Fast prediction
add_box(8.0, y_bot, w, h, TEAL_MED,
        "Fast prediction\nŷ*")

# Arrows (bottom row)
add_arrow((2.0 + w, y_bot + h/2),
          (5.0,      y_bot + h/2))
add_arrow((5.0 + w, y_bot + h/2),
          (8.0,      y_bot + h/2))

ax.text(5.0, 0.4,
        "Online: use surrogate for fast predictions instead of expensive model",
        ha="center", va="center", fontsize=9)

plt.tight_layout()
plt.show()

