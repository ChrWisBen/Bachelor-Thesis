# -*- coding: utf-8 -*-
"""
Combined plot:
Plots P₁,₁ ... P₁,₃, P₂,₁ ... P₂,₃, P₃,₁ ... P₃,₃, P₄,₁ ... P₄,₃
and a single P₄,₄ (H=0) on one figure.
All digits in labels are shown as Unicode subscripts.
"""

import math
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

# ---------------- Colors ----------------
TEAL_LIGHT = "#B8D1CA"
TEAL_MID   = "#5F938C"
TEAL_DARK  = "#296872"
ORANGE     = "#F79433"
RED        = "#C74A3A"
PURPLE     = "#78506E"

# Explicit wall color mapping (by wall ID):
WALL_COLOR_BY_NAME = {
    "W₁": TEAL_DARK,     # RC
    "W₄": TEAL_DARK,     # RC
    "W₂": TEAL_DARK,     # RC
    "W₃": TEAL_DARK,     # RC
}

# Display names for walls (what is shown in the figure):
WALL_DISPLAY_BY_NAME = {
    "W₁": "W₁-RC",
    "W₄": "W₄-RC",
    "W₂": "W₂-RC",
    "W₃": "W₃-RC",
}

# Point colors
P1_COLOR = ORANGE
P2_COLOR = RED
P3_COLOR = PURPLE
P4_COLOR = TEAL_DARK

# --------------- Geometry ---------------
walls = [
    ("W₁", "V", (1, 2), (1, 8)),   # vertical  (length 6)
    ("W₂", "V", (7, 3), (7, 7)),   # vertical  (length 4)
    ("W₃", "H", (3, 9), (7, 9)),   # horizontal (length 4)
    ("W₄", "H", (2, 1), (8, 1)),   # horizontal (length 6)
]

def wall_length(w):
    _, _, (x1, y1), (x2, y2) = w
    return math.hypot(x2 - x1, y2 - y1)

def wall_centroid(w):
    _, _, (x1, y1), (x2, y2) = w
    return ((x1 + x2) / 2, (y1 + y2) / 2)

centroids_wall = {w[0]: wall_centroid(w) for w in walls}

# --------------- K tables (4 "cases") ---------------
# H=5
K_TABLES_1 = {
    "k_capacity_CLT_V25":   {2:29, 4:99, 6:162, 8:230, 10:298},         # CLT
    "k_capacity_RC_V25":    {2:168, 4:382, 6:596, 8:809, 10:1023},      # RC
    "k_capacity_CLT_V50":   {2:4,  4:74, 6:137, 8:205, 10:273},         # CLT
    "k_capacity_RC_V50":    {2:133, 4:346, 6:560, 8:773, 10:987},       # RC
    "k_capacity_CLT_V75":   {2:0,  4:49, 6:112, 8:180, 10:248},         # CLT
    "k_capacity_RC_V75":    {2:97,  4:310, 6:524, 8:737, 10:951},       # RC
    "k_capacity_CLT_V100":  {2:0,  4:18, 6:87,  8:155, 10:223},         # CLT
    "k_capacity_RC_V100":   {2:62, 4:274, 6:488, 8:702, 10:915},        # RC
}

# H=10
K_TABLES_2 = {
    "k_capacity_CLT_V25":   {2:29, 4:99, 6:162, 8:230, 10:298},         # CLT
    "k_capacity_RC_V25":    {2:132, 4:346, 6:560, 8:773, 10:987},       # RC
    "k_capacity_CLT_V50":   {2:4,  4:74, 6:137, 8:205, 10:273},         # CLT
    "k_capacity_RC_V50":    {2:62, 4:274, 6:488, 8:702, 10:915},        # RC
    "k_capacity_CLT_V75":   {2:0,  4:48, 6:112, 8:179, 10:248},         # CLT
    "k_capacity_RC_V75":    {2:62, 4:204, 6:416, 8:630, 10:843},        # RC
    "k_capacity_CLT_V100":  {2:0,  4:18, 6:87,  8:155, 10:223},         # CLT
    "k_capacity_RC_V100":   {2:0,  4:132, 6:346, 8:560, 10:772},        # RC
}

# H=20
K_TABLES_3 = {
    "k_capacity_CLT_V25":   {2:0, 4:73,  6:137,   8:205,   10:273},     # CLT
    "k_capacity_RC_V25":    {2:0, 4:132, 6:346,   8:560,   10:772},     # RC
    "k_capacity_CLT_V50":   {2:0, 4:73,  6:137,   8:205,   10:273},     # CLT
    "k_capacity_RC_V50":    {2:0, 4:132, 6:346,   8:560,   10:772},     # RC
    "k_capacity_CLT_V75":   {2:0, 4:48.11, 6:111.5, 8:179.71, 10:247.84}, # CLT
    "k_capacity_RC_V75":    {2:0, 4:0,    6:203,   8:416,   10:630},    # RC
    "k_capacity_CLT_V100":  {2:0, 4:18,  6:86,    8:155,   10:223},     # CLT
    "k_capacity_RC_V100":   {2:0, 4:0,   6:59,    8:273,   10:486},     # RC
}

# Fourth table (base / H=0 case, only used for the extra P₄,₄ point)
K_TABLES_4 = {
    "k_capacity_H0_CLT": {2:4.39, 4:73.72, 6:137, 8:205, 10:273},  # CLT
    "k_capacity_H0_RC":  {2:204, 4:418,   6:631, 8:845, 10:1058},  # RC
}

SCENARIOS = [
    ("1", K_TABLES_1, "H=5"),
    ("2", K_TABLES_2, "H=10"),
    ("3", K_TABLES_3, "H=20"),
    ("4", K_TABLES_4, "H=0"),  # only used for the extra P₄,₄ point
]

# --------------- k-choice per P (metric choice) ---------------
CENTROID_K_CHOICE = {
    # P₁: capacity RC at V=25%
    "P₁": {"W₁":"k_capacity_RC_V25", "W₂":"k_capacity_RC_V25",
           "W₃":"k_capacity_RC_V25", "W₄":"k_capacity_RC_V25"},
    # P₂: capacity RC at V=50%
    "P₂": {"W₁":"k_capacity_RC_V50", "W₂":"k_capacity_RC_V50",
           "W₃":"k_capacity_RC_V50", "W₄":"k_capacity_RC_V50"},
    # P₃: capacity RC at V=75%
    "P₃": {"W₁":"k_capacity_RC_V75", "W₂":"k_capacity_RC_V75",
           "W₃":"k_capacity_RC_V75", "W₄":"k_capacity_RC_V75"},
    # P₄: capacity RC at V=100% (for H=5,10,20)
    "P₄": {"W₁":"k_capacity_RC_V100", "W₂":"k_capacity_RC_V100",
           "W₃":"k_capacity_RC_V100", "W₄":"k_capacity_RC_V100"},
}

# --------------- Subscript helper ---------------
_SUBS_MAP = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
def subscript_digits(text: str) -> str:
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
    H = [w for w in walls if w[1] == "H"]
    V = [w for w in walls if w[1] == "V"]

    sum_k_h  = sum(kv[w[0]] for w in H)
    sum_ky_h = sum(kv[w[0]] * centroids_wall[w[0]][1] for w in H)
    x_c_raw  = (sum_ky_h / sum_k_h) if sum_k_h != 0 else 5.0

    sum_k_v  = sum(kv[w[0]] for w in V)
    sum_kx_v = sum(kv[w[0]] * centroids_wall[w[0]][0] for w in V)
    y_c_raw  = (sum_kx_v / sum_k_v) if sum_k_v != 0 else 5.0

    return y_c_raw, x_c_raw  # swapped

# --------------- Compute all required points ---------------
points = []

# For P₁–P₄ we use the first three scenarios (H=5,10,20)
SCENARIOS_123 = SCENARIOS[:3]

# P₁ over 3 scenarios
for scen_tag, Kt, _title in SCENARIOS_123:
    kv = k_values_for_choice(CENTROID_K_CHOICE["P₁"], Kt)
    xc, yc = compute_centroid_custom(kv)
    points.append((f"P₁,{subscript_digits(scen_tag)}", P1_COLOR, (xc, yc)))

# P₂ over 3 scenarios
for scen_tag, Kt, _title in SCENARIOS_123:
    kv = k_values_for_choice(CENTROID_K_CHOICE["P₂"], Kt)
    xc, yc = compute_centroid_custom(kv)
    points.append((f"P₂,{subscript_digits(scen_tag)}", P2_COLOR, (xc, yc)))

# P₃ over 3 scenarios
for scen_tag, Kt, _title in SCENARIOS_123:
    kv = k_values_for_choice(CENTROID_K_CHOICE["P₃"], Kt)
    xc, yc = compute_centroid_custom(kv)
    points.append((f"P₃,{subscript_digits(scen_tag)}", P3_COLOR, (xc, yc)))

# P₄ over 3 scenarios (V=100%)
for scen_tag, Kt, _title in SCENARIOS_123:
    kv = k_values_for_choice(CENTROID_K_CHOICE["P₄"], Kt)
    xc, yc = compute_centroid_custom(kv)
    points.append((f"P₄,{subscript_digits(scen_tag)}", P4_COLOR, (xc, yc)))  # <-- fixed line

# Extra "last point": P₄,₄ based on H=0 table (K_TABLES_4)
kv_p4_base = k_values_for_choice(
    {"W₁":"k_capacity_H0_RC", "W₂":"k_capacity_H0_RC",
     "W₃":"k_capacity_H0_RC", "W₄":"k_capacity_H0_RC"},
    K_TABLES_4
)
xc4, yc4 = compute_centroid_custom(kv_p4_base)
points.append((f"P₄,{subscript_digits('4')}", P4_COLOR, (xc4, yc4)))

# --------------- Overlap helpers ---------------
def bboxes_overlap(b1, b2, pad=2):
    return not (
        b1.x1 + pad < b2.x0 or
        b1.x0 - pad > b2.x1 or
        b1.y1 + pad < b2.y0 or
        b1.y0 - pad > b2.y1
    )

def place_label_no_overlap(ax, fig, x, y, text, color,
                           existing_label_bboxes, point_bboxes, fontsize=9):
    renderer = fig.canvas.get_renderer()

    offsets = []
    for r in [0.10, 0.18, 0.28, 0.40, 0.55, 0.75, 1.0]:
        for dx, dy in [
            ( r,  r), ( r, 0), ( r,-r),
            (0,  r), (0,-r),
            (-r, r), (-r,0), (-r,-r)
        ]:
            offsets.append((dx, dy))

    for dx, dy in offsets:
        t = ax.text(x + dx, y + dy, subscript_digits(text),
                    color=color, fontsize=fontsize,
                    ha="left", va="bottom")
        fig.canvas.draw()
        bb = t.get_window_extent(renderer=renderer)

        overlaps_label = any(bboxes_overlap(bb, prev) for prev in existing_label_bboxes)
        overlaps_point = any(bboxes_overlap(bb, pb) for pb in point_bboxes)

        if (not overlaps_label) and (not overlaps_point):
            existing_label_bboxes.append(bb)
            return t
        t.remove()

    t = ax.text(x + 0.10, y + 0.10, subscript_digits(text),
                color=color, fontsize=fontsize,
                ha="left", va="bottom")
    fig.canvas.draw()
    existing_label_bboxes.append(t.get_window_extent(renderer=renderer))
    return t

def place_wall_label_with_arrow(ax, fig, wall, color,
                                existing_label_bboxes, point_bboxes,
                                base_offset=0.55, fontsize=10):
    renderer = fig.canvas.get_renderer()
    name, _, (x1, y1), (x2, y2) = wall
    cx, cy = wall_centroid(wall)

    dx, dy = (x2 - x1), (y2 - y1)
    L = math.hypot(dx, dy) if math.hypot(dx, dy) != 0 else 1.0
    nx, ny = (-dy / L), (dx / L)

    if name in ("W₁", "W₂"):
        side_order = [-1]
    else:
        side_order = [1, -1]

    display_text = WALL_DISPLAY_BY_NAME.get(name, name)

    candidates = []
    for side in side_order:
        for scale in [1.0, 1.4, 1.8, 2.4]:
            candidates.append((side * nx * base_offset * scale,
                               side * ny * base_offset * scale))

    for ox, oy in candidates:
        ann = ax.annotate(
            subscript_digits(display_text),
            xy=(cx, cy), xycoords="data",
            xytext=(cx + ox, cy + oy), textcoords="data",
            fontsize=fontsize, color=color,
            ha="center", va="center",
            arrowprops=dict(
                arrowstyle="-",   # line only, no arrow head
                lw=0.8, color=color,
                shrinkA=2, shrinkB=2,
                mutation_scale=8
            ),
            zorder=6
        )

        fig.canvas.draw()
        bb = ann.get_window_extent(renderer=renderer)

        overlaps_label = any(bboxes_overlap(bb, prev) for prev in existing_label_bboxes)
        overlaps_point = any(bboxes_overlap(bb, pb) for pb in point_bboxes)

        if (not overlaps_label) and (not overlaps_point):
            existing_label_bboxes.append(bb)
            return ann

        ann.remove()

    side = side_order[0]
    ann = ax.annotate(
        subscript_digits(display_text),
        xy=(cx, cy), xycoords="data",
        xytext=(cx + side * nx * base_offset, cy + side * ny * base_offset),
        textcoords="data",
        fontsize=fontsize, color=color,
        ha="center", va="center",
        arrowprops=dict(
            arrowstyle="-",   # line only, no arrow head
            lw=0.8, color=color,
            shrinkA=2, shrinkB=2,
            mutation_scale=8
        ),
        zorder=6
    )
    fig.canvas.draw()
    existing_label_bboxes.append(ann.get_window_extent(renderer=renderer))
    return ann

# --------------- Plot ---------------
def plot_combined():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')

    ax.set_title("Rotational Centres, Capacity Approach, RC - Differeing V, H",
                 fontsize=13, pad=10)

    # Walls with explicit colors
    for w in walls:
        name, _, (x1, y1), (x2, y2) = w
        color = WALL_COLOR_BY_NAME.get(name, TEAL_MID)
        ax.plot([x1, x2], [y1, y2], linewidth=4, color=color, zorder=1)

    # Small dots for points
    marker_size = 5
    for _, col, (x, y) in points:
        ax.plot(x, y, 'o', ms=marker_size, color=col, zorder=5)

    # Build exclusion bboxes around each dot (display coords)
    fig.canvas.draw()
    dpi = fig.dpi
    radius_px = (marker_size / 2) * dpi / 72.0
    pad_px = 2.0

    point_bboxes = []
    for _, _, (x, y) in points:
        x_disp, y_disp = ax.transData.transform((x, y))
        r = radius_px + pad_px
        pb = Bbox.from_extents(x_disp - r, y_disp - r, x_disp + r, y_disp + r)
        point_bboxes.append(pb)

    existing_label_bboxes = []

    # Wall labels offset with lines (color follows wall)
    for w in walls:
        name = w[0]
        color = WALL_COLOR_BY_NAME.get(name, TEAL_MID)
        place_wall_label_with_arrow(
            ax, fig, w, color,
            existing_label_bboxes, point_bboxes,
            base_offset=0.55, fontsize=10
        )

    # Point labels with no-overlap logic
    text_handles = []
    for lbl, col, (x, y) in points:
        t = place_label_no_overlap(
            ax, fig, x, y, lbl, col,
            existing_label_bboxes, point_bboxes, fontsize=9
        )
        text_handles.append((t, x, y, col))

    # Lines (no arrowheads) from point label -> dot
    for t, x_dot, y_dot, col in text_handles:
        x_txt, y_txt = t.get_position()
        ax.annotate(
            "",
            xy=(x_dot, y_dot),
            xytext=(x_txt, y_txt),
            arrowprops=dict(
                arrowstyle="-",   # line only, no arrow head
                lw=0.8, color=col,
                shrinkA=2, shrinkB=2,
                mutation_scale=8
            ),
            zorder=4
        )

    # Reference arrows (wx, wy) kept as arrows
    ax.annotate(
        "",
        xy=(0.5, 5), xytext=(0, 5),
        arrowprops=dict(arrowstyle="->", lw=1.2, color=TEAL_MID),
        zorder=3
    )
    ax.text(0.25, 5.2, "wx", ha="center", va="bottom",
            fontsize=11, color=TEAL_MID)

    ax.annotate(
        "",
        xy=(5, 0.5), xytext=(5, 0),
        arrowprops=dict(arrowstyle="->", lw=1.2, color=TEAL_MID),
        zorder=3
    )
    ax.text(5.2, 0.25, "wy", ha="left", va="center",
            fontsize=11, color=TEAL_MID)

    ax.grid(True)
    plt.tight_layout()
    plt.savefig("rotational_centres_Capacity_Many.svg", format="svg")
    plt.show()

plot_combined()


