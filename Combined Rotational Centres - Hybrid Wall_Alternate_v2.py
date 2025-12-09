# -*- coding: utf-8 -*-
"""
Combined plot:
Plots P₁,₁ P₁,₂ P₁,₃  P₂,₁ P₂,₂ P₂,₃  P₃  P₄ on one figure.
All digits in labels are shown as Unicode subscripts.

Material-specific k-variants per table:
- k_capacity_N0_CLT   (capacity with N=0, CLT)
- k_capacity_N200_RC  (capacity with N=200, RC)
- k_unit_CLT / k_unit_RC
- k_shear_CLT / k_shear_RC
- k_bend_CLT / k_bend_RC

Update per request:
- W₁ and W₂ are DARK TEAL (#296872)
- W₃ and W₄ are LIGHT TEAL (#B8D1CA)
"""

import math
import matplotlib.pyplot as plt

# ---------------- Colors ----------------
TEAL_LIGHT = "#B8D1CA"
TEAL_MID   = "#5F938C"
TEAL_DARK  = "#296872"
ORANGE     = "#F79433"

# Explicit wall color mapping (overrides cycling):
WALL_COLOR_BY_NAME = {
    "W₁": ORANGE,  # dark teal
    "W₂": TEAL_DARK,  # dark teal
    "W₃": TEAL_DARK, # light teal
    "W₄": ORANGE, # light teal
}

# Point colors
P1_COLOR = ORANGE
P2_COLOR = TEAL_LIGHT
P3_COLOR = TEAL_DARK
P4_COLOR = TEAL_MID

# --------------- Geometry ---------------
walls = [
    ("W₁", "V", (1, 2), (1, 8)),   # vertical  (length 6)
    ("W₂", "V", (7, 3), (7, 7)),   # vertical  (length 4)
    ("W₃", "H", (3, 9), (7, 9)),   # horizontal (length 4)
    ("W₄", "H", (2, 1), (8, 1)),   # horizontal (length 6)
]

def wall_length(w):
    _, _, (x1,y1), (x2,y2) = w
    return math.hypot(x2 - x1, y2 - y1)

def wall_centroid(w):
    _, _, (x1,y1), (x2,y2) = w
    return ((x1 + x2)/2, (y1 + y2)/2)

centroids_wall = {w[0]: wall_centroid(w) for w in walls}

# --------------- Three K tables (your three cases) ---------------
# H=4
K_TABLES_1 = {
    "k_capacity_N0_CLT":   {2:11982, 4:38730, 6:65578, 8:91565, 10:117013},   # CLT
    "k_capacity_N200_RC":  {2:85054, 4:3095360, 6:4643040, 8:6190720, 10:7738400},  # RC
    "k_unit_CLT":          {2:18813, 4:45389, 6:70788, 8:95714, 10:120429},
    "k_unit_RC":           {2:196442, 4:1138149, 6:2631681, 8:4329444, 10:6068652},
    "k_shear_CLT":         {2:96000, 4:192000, 6:288000, 8:384000, 10:480000},
    "k_shear_RC":          {2:1548000, 4:3096000, 6:4644000, 8:6192000, 10:7740000},
    "k_bend_CLT":          {2:82500, 4:660000, 6:2227500, 8:5280000, 10:10312500},
    "k_bend_RC":           {2:225000, 4:1800000, 6:6075000, 8:14400000, 10:28125000},
}

# H=10
K_TABLES_2 = {
    "k_capacity_N0_CLT":   {2:2077, 4:10138, 6:20737, 8:31681, 10:42468},      # CLT
    "k_capacity_N200_RC":  {2:8871, 4:233651, 6:1857216, 8:2476288, 10:3095360},    # RC
    "k_unit_CLT":          {2:3425, 4:13339, 6:24266, 8:34959, 10:45389},
    "k_unit_RC":           {2:14073, 4:105394, 6:321496, 8:671637, 10:1138149},
    "k_shear_CLT":         {2:38400, 4:76800, 6:115200, 8:153600, 10:192000},
    "k_shear_RC":          {2:619200, 4:1238400, 6:1857600, 8:2476800, 10:3096000},
    "k_bend_CLT":          {2:5280, 4:42240, 6:142560, 8:337920, 10:660000},
    "k_bend_RC":           {2:14400, 4:115200, 6:388800, 8:921600, 10:1800000},
}

# H=20
K_TABLES_3 = {
    "k_capacity_N0_CLT":   {2:404, 4:2586, 6:6555, 8:11520, 10:16888},         # CLT
    "k_capacity_N200_RC":  {2:1378, 4:19910, 6:928608, 8:1238144, 10:1547680},          # RC
    "k_unit_CLT":          {2:581, 4:3425, 6:8032, 8:13339, 10:18813},
    "k_unit_RC":           {2:1790, 4:14073, 6:46183, 8:105394, 10:196442},
    "k_shear_CLT":         {2:19200, 4:38400, 6:57600, 8:76800, 10:96000},
    "k_shear_RC":          {2:309600, 4:619200, 6:928800, 8:1238400, 10:1548000},
    "k_bend_CLT":          {2:660, 4:5280, 6:17820, 8:42240, 10:82500},
    "k_bend_RC":           {2:1800, 4:14400, 6:48600, 8:115200, 10:225000},
}

SCENARIOS = [
    ("1", K_TABLES_1, "H=4"),
    ("2", K_TABLES_2, "H=10"),
    ("3", K_TABLES_3, "H=20"),
]

# --------------- k-choice per P (uses the exact keys above) ---------------
CENTROID_K_CHOICE = {
    # P₁: RC on verticals (cap N=200), CLT on horizontals (cap N=0)
    "P₁": {"W₁":"k_capacity_N0_CLT", "W₂":"k_capacity_N200_RC",
           "W₃":"k_capacity_N200_RC",  "W₄":"k_capacity_N0_CLT"},
    # P₂: unit (mix shown; CLT/RC are different per your tables)
    "P₂": {"W₁":"k_unit_CLT", "W₂":"k_unit_RC", "W₃":"k_unit_RC", "W₄":"k_unit_CLT"},
    # P₃: shear mix as specified
    "P₃": {"W₁":"k_shear_CLT", "W₂":"k_shear_RC", "W₃":"k_shear_RC", "W₄":"k_shear_CLT"},
    # P₄: bending mix as specified
    "P₄": {"W₁":"k_bend_CLT", "W₂":"k_bend_RC", "W₃":"k_bend_RC", "W₄":"k_bend_CLT"},
}

# --------------- Subscript helper ---------------
_SUBS_MAP = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
def subscript_digits(text: str) -> str:
    """Convert all ASCII digits in text to Unicode subscript digits."""
    return str(text).translate(_SUBS_MAP)

# --------------- Helpers ---------------
def k_values_for_choice(choice_map, K_TABLES):
    kv = {}
    for w in walls:
        name = w[0]
        L = round(wall_length(w))
        metric = choice_map[name]
        kv[name] = K_TABLES[metric][L]
    return kv

def compute_centroid_custom(kv):
    """x_c from H using y, y_c from V using x; then swap."""
    H = [w for w in walls if w[1] == "H"]
    V = [w for w in walls if w[1] == "V"]

    sum_k_h  = sum(kv[w[0]] for w in H)
    sum_ky_h = sum(kv[w[0]] * centroids_wall[w[0]][1] for w in H)
    x_c_raw  = (sum_ky_h / sum_k_h) if sum_k_h != 0 else 5.0

    sum_k_v  = sum(kv[w[0]] for w in V)
    sum_kx_v = sum(kv[w[0]] * centroids_wall[w[0]][0] for w in V)
    y_c_raw  = (sum_kx_v / sum_k_v) if sum_k_v != 0 else 5.0

    x_c = y_c_raw
    y_c = x_c_raw
    return x_c, y_c

# --------------- Compute all required points ---------------
points = []  # list of (label, color, (x,y))

# P1 across the 3 scenarios
for scen_tag, Kt, _title in SCENARIOS:
    kv = k_values_for_choice(CENTROID_K_CHOICE["P₁"], Kt)
    xc, yc = compute_centroid_custom(kv)
    label = f"P₁,{subscript_digits(scen_tag)}"  # e.g., P₁,₁
    points.append((label, P1_COLOR, (xc, yc)))

# P2 across the 3 scenarios
for scen_tag, Kt, _title in SCENARIOS:
    kv = k_values_for_choice(CENTROID_K_CHOICE["P₂"], Kt)
    xc, yc = compute_centroid_custom(kv)
    label = f"P₂,{subscript_digits(scen_tag)}"  # e.g., P₂,₁
    points.append((label, P2_COLOR, (xc, yc)))

# P3 (compute once)
kv_p3 = k_values_for_choice(CENTROID_K_CHOICE["P₃"], K_TABLES_1)
xc3, yc3 = compute_centroid_custom(kv_p3)
points.append(("P₃", P3_COLOR, (xc3, yc3)))

# P4 (compute once)
kv_p4 = k_values_for_choice(CENTROID_K_CHOICE["P₄"], K_TABLES_1)
xc4, yc4 = compute_centroid_custom(kv_p4)
points.append(("P₄", P4_COLOR, (xc4, yc4)))

# --------------- Plot ---------------
def plot_combined():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect('equal')

    # Title
    ax.set_title("Rotational Centres, Hybrid Approach, Variation 1", fontsize=13, pad=10)

    # Walls with explicit colors per name
    for (name, _, (x1, y1), (x2, y2)) in walls:
        color = WALL_COLOR_BY_NAME.get(name, TEAL_MID)
        ax.plot([x1, x2], [y1, y2], linewidth=4, color=color)
        cx, cy = centroids_wall[name]
        ax.text(cx, cy, subscript_digits(name), ha='center', va='center', fontsize=10, color=color)

    # Points with labels
    for lbl, col, (x, y) in points:
        ax.plot(x, y, 'o', ms=9, color=col)
        ax.text(x + 0.12, y + 0.12, subscript_digits(lbl), color=col, fontsize=9)

    ax.grid(True)
    plt.tight_layout()
    # Save as SVG for vector output
    plt.savefig("rotational_centres_Hybrid_Alternate.svg", format="svg")
    plt.show()

plot_combined()

