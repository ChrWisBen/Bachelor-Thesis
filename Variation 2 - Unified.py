# -*- coding: utf-8 -*-
"""
Combined plot (original kept but no longer called at the bottom):
Plots P₁,₁ P₁,₂ P₁,₃  P₂,₁ P₂,₂ P₂,₃  P₃  P₄ on one figure.
All digits in labels are shown as Unicode subscripts.
"""

import math
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.collections import LineCollection   # NEW
import matplotlib.colors as mcolors                 # NEW
import numpy as np                                  # NEW

# ---------------- Colors ----------------
TEAL_LIGHT = "#B8D1CA"
TEAL_MID   = "#5F938C"
TEAL_DARK  = "#296872"
ORANGE     = "#F79433"
RED        = "#C74A3A"

# Explicit wall color mapping (by wall ID):
WALL_COLOR_BY_NAME = {
    "W₁": ORANGE,     # CLT
    "W₄": ORANGE,        # CLT
    "W₂": TEAL_DARK,     # RC
    "W₃": TEAL_DARK,        # RC
}

# Display names for walls (what is shown in the figure):
WALL_DISPLAY_BY_NAME = {
    "W₁": "W₁-CLT",
    "W₄": "W₄-CLT",
    "W₂": "W₂-RC",
    "W₃": "W₃-RC",
}

# Point colors
P1_COLOR = ORANGE
P2_COLOR = RED
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
    _, _, (x1, y1), (x2, y2) = w
    return math.hypot(x2 - x1, y2 - y1)

def wall_centroid(w):
    _, _, (x1, y1), (x2, y2) = w
    return ((x1 + x2) / 2, (y1 + y2) / 2)

centroids_wall = {w[0]: wall_centroid(w) for w in walls}

# --------------- Three K tables (your three cases) ---------------
# H=4
K_TABLES_1 = {
    "k_capacity_N0_CLT":   {2:4.39, 4:73.72, 6:137, 8:205, 10:272},   # CLT
    "k_capacity_N200_RC":  {2:204, 4:418, 6:631, 8:845, 10:1058},  # RC
    "k_unit_CLT":          {2:11982, 4:38730, 6:65578, 8:91565, 10:117013},
    "k_unit_RC":           {2:85054, 4:3095360, 6:4643040, 8:6190720, 10:7738400},
    "k_shear_CLT":         {2:96000, 4:192000, 6:288000, 8:384000, 10:480000},
    "k_shear_RC":          {2:1548000, 4:3096000, 6:4644000, 8:6192000, 10:7740000},
    "k_bend_CLT":          {2:82500, 4:660000, 6:2227500, 8:5280000, 10:10312500},
    "k_bend_RC":           {2:225000, 4:1800000, 6:6075000, 8:14400000, 10:28125000},
}

# H=10
K_TABLES_2 = {
    "k_capacity_N0_CLT":   {2:4, 4:74, 6:137, 8:205, 10:273},      # CLT
    "k_capacity_N200_RC":  {2:132, 4:346, 6:560, 8:773, 10:987},    # RC
    "k_unit_CLT":          {2:2077, 4:10138, 6:20737, 8:31681, 10:42468},
    "k_unit_RC":           {2:8871, 4:233651, 6:1857216, 8:2476288, 10:3095360},
    "k_shear_CLT":         {2:38400, 4:76800, 6:115200, 8:153600, 10:192000},
    "k_shear_RC":          {2:619200, 4:1238400, 6:1857600, 8:2476800, 10:3096000},
    "k_bend_CLT":          {2:5280, 4:42240, 6:142560, 8:337920, 10:660000},
    "k_bend_RC":           {2:14400, 4:115200, 6:388800, 8:921600, 10:1800000},
}

# H=20
K_TABLES_3 = {
    "k_capacity_N0_CLT":   {2:3, 4:74, 6:137, 8:205, 10:273},         # CLT
    "k_capacity_N200_RC":  {2:62, 4:274, 6:488, 8:702, 10:915},       # RC
    "k_unit_CLT":          {2:404, 4:2586, 6:6555, 8:11520, 10:16888},
    "k_unit_RC":           {2:1378, 4:19910, 6:928608, 8:1238144, 10:1547680},
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
    "P₁": {"W₁":"k_capacity_N0_CLT", "W₂":"k_capacity_N200_RC",
           "W₃":"k_capacity_N200_RC",  "W₄":"k_capacity_N0_CLT"},
    "P₂": {"W₁":"k_unit_CLT", "W₂":"k_unit_RC", "W₃":"k_unit_RC", "W₄":"k_unit_CLT"},
    "P₃": {"W₁":"k_shear_CLT", "W₂":"k_shear_RC", "W₃":"k_shear_RC", "W₄":"k_shear_CLT"},
    "P₄": {"W₁":"k_bend_CLT", "W₂":"k_bend_RC", "W₃":"k_bend_RC", "W₄":"k_bend_CLT"},
}

# --------------- Subscript helper ---------------
_SUBS_MAP = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
def subscript_digits(text: str) -> str:
    return str(text).translate(_SUBS_MAP)

# NEW: reverse for going from "P₁,₁" -> "P1,1"
_SUBSCRIPT_DIGITS = "₀₁₂₃₄₅₆₇₈₉"            # NEW
_PLAIN_DIGITS     = "0123456789"            # NEW
_REV_SUBS_MAP = {s: p for p, s in zip(_PLAIN_DIGITS, _SUBSCRIPT_DIGITS)}  # NEW

def desubscript_digits(text: str) -> str:    # NEW
    return "".join(_REV_SUBS_MAP.get(ch, ch) for ch in text)

def label_to_plain(lbl: str) -> str:        # NEW
    if not lbl.startswith("P"):
        return lbl
    return "P" + desubscript_digits(lbl[1:])

def plain_to_display(lbl_plain: str) -> str:  # NEW, e.g. "P1,2" -> "P₁,₂"
    if not lbl_plain.startswith("P"):
        return lbl_plain
    return "P" + subscript_digits(lbl_plain[1:])

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

for scen_tag, Kt, _title in SCENARIOS:
    kv = k_values_for_choice(CENTROID_K_CHOICE["P₁"], Kt)
    xc, yc = compute_centroid_custom(kv)
    points.append((f"P₁,{subscript_digits(scen_tag)}", P1_COLOR, (xc, yc)))

for scen_tag, Kt, _title in SCENARIOS:
    kv = k_values_for_choice(CENTROID_K_CHOICE["P₂"], Kt)
    xc, yc = compute_centroid_custom(kv)
    points.append((f"P₂,{subscript_digits(scen_tag)}", P2_COLOR, (xc, yc)))

kv_p3 = k_values_for_choice(CENTROID_K_CHOICE["P₃"], K_TABLES_1)
xc3, yc3 = compute_centroid_custom(kv_p3)
points.append(("P₃", P3_COLOR, (xc3, yc3)))

kv_p4 = k_values_for_choice(CENTROID_K_CHOICE["P₄"], K_TABLES_1)
xc4, yc4 = compute_centroid_custom(kv_p4)
points.append(("P₄", P4_COLOR, (xc4, yc4)))

# NEW: map from "plain" labels like "P1,1", "P2,3", "P3" to actual data
points_map_plain = {}  # key: "P1,1" etc -> (display_label, color, (x,y))  # NEW
for lbl, col, xy in points:                                                # NEW
    key = label_to_plain(lbl)                                              # NEW
    points_map_plain[key] = (lbl, col, xy)                                 # NEW

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

# --------------- Gradient line helper (NEW) ---------------
def draw_gradient_line(ax, p_start, p_end, color_start, color_end,
                       linewidth=2.0, n_segments=200):
    """
    Approximate a gradient line from p_start to p_end, going from color_start
    to color_end using many small segments.
    """
    x0, y0 = p_start
    x1, y1 = p_end

    xs = np.linspace(x0, x1, n_segments + 1)
    ys = np.linspace(y0, y1, n_segments + 1)

    segments = [
        [(xs[i], ys[i]), (xs[i + 1], ys[i + 1])]
        for i in range(n_segments)
    ]

    c1 = np.array(mcolors.to_rgb(color_start))
    c2 = np.array(mcolors.to_rgb(color_end))
    ts = np.linspace(0.0, 1.0, n_segments)

    colors = [c1 * (1 - t) + c2 * t for t in ts]

    lc = LineCollection(segments, colors=colors, linewidth=linewidth, zorder=2)
    ax.add_collection(lc)

# --------------- Base drawing for any pair (NEW) ---------------
def plot_pair(pair_plain, filename, title_text):
    """
    pair_plain: tuple of two plain labels, e.g. ("P1,1", "P2,1")
    filename: output SVG filename
    title_text: full title to show in the figure
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')

    ax.set_title(title_text, fontsize=13, pad=10)

    # Walls
    for w in walls:
        name, _, (x1, y1), (x2, y2) = w
        color = WALL_COLOR_BY_NAME.get(name, TEAL_MID)
        ax.plot([x1, x2], [y1, y2], linewidth=4, color=color, zorder=1)

    # Select the two points
    selected_points = []
    for key in pair_plain:
        lbl, col, xy = points_map_plain[key]
        selected_points.append((lbl, col, xy))

    # Plot points
    marker_size = 5
    for _, col, (x, y) in selected_points:
        ax.plot(x, y, 'o', ms=marker_size, color=col, zorder=5)

    # Build exclusion bboxes around each dot
    fig.canvas.draw()
    dpi = fig.dpi
    radius_px = (marker_size / 2) * dpi / 72.0
    pad_px = 2.0

    point_bboxes = []
    for _, _, (x, y) in selected_points:
        x_disp, y_disp = ax.transData.transform((x, y))
        r = radius_px + pad_px
        pb = Bbox.from_extents(x_disp - r, y_disp - r, x_disp + r, y_disp + r)
        point_bboxes.append(pb)

    existing_label_bboxes = []

    # Wall labels
    for w in walls:
        name = w[0]
        color = WALL_COLOR_BY_NAME.get(name, TEAL_MID)
        place_wall_label_with_arrow(
            ax, fig, w, color,
            existing_label_bboxes, point_bboxes,
            base_offset=0.55, fontsize=10
        )

    # Point labels
    text_handles = []
    for lbl, col, (x, y) in selected_points:
        t = place_label_no_overlap(
            ax, fig, x, y, lbl, col,
            existing_label_bboxes, point_bboxes, fontsize=9
        )
        text_handles.append((t, x, y, col))

    # Lines from point label -> dot
    for t, x_dot, y_dot, col in text_handles:
        x_txt, y_txt = t.get_position()
        ax.annotate(
            "",
            xy=(x_dot, y_dot),
            xytext=(x_txt, y_txt),
            arrowprops=dict(
                arrowstyle="-",
                lw=0.8, color=col,
                shrinkA=2, shrinkB=2,
                mutation_scale=8
            ),
            zorder=4
        )

    # Gradient line from first point to second
    (_, col1, (x1, y1)) = selected_points[0]
    (_, col2, (x2, y2)) = selected_points[1]
    draw_gradient_line(ax, (x1, y1), (x2, y2), col1, col2,
                       linewidth=2.0, n_segments=300)

    # Reference arrows (wx, wy)
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
    plt.savefig(filename, format="svg")
    plt.close(fig)

# --------------- Original combined plot (kept for reference) ---------------
def plot_combined():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect('equal')

    ax.set_title("Rotational Centres, Unified Approach, Variation 2",
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
    plt.savefig("rotational_centres_Unified_Variation2.svg", format="svg")
    plt.close(fig)

# --------------- Make the three requested SVGs (NEW) ---------------
if __name__ == "__main__":
    # 1) P1,1 and P2,1  (H=4)
    pair1 = ("P1,1", "P2,1")
    title1 = "Variation 2, H(v) = 0m – " + plain_to_display("P1,1") + "–" + plain_to_display("P2,1")
    plot_pair(pair1, "rotational_centres_P1_1_P2_1_Variation2.svg", title1)

    # 2) P1,2 and P2,2  (H=10)
    pair2 = ("P1,2", "P2,2")
    title2 = "Variation 2, H(v) = 5m – " + plain_to_display("P1,2") + "–" + plain_to_display("P2,2")
    plot_pair(pair2, "rotational_centres_P1_2_P2_2_Variation2.svg", title2)

    # 3) P1,3 and P2,3  (H=20)
    pair3 = ("P1,3", "P2,3")
    title3 = "Variation 2, H(v) = 10m – " + plain_to_display("P1,3") + "–" + plain_to_display("P2,3")
    plot_pair(pair3, "rotational_centres_P1_3_P2_3_Variation2.svg", title3)


