# -*- coding: utf-8 -*-
"""
Four centroid variants on one plot, with multiple scenarios.

Now:
- P₁, P₂, P₃ each have 4 points: P₁,₁ … P₁,₄, P₂,₁ …, P₃,₁ …
- P₄ has a single point based on the last k-table scenario.

K-tables are organised by "height" keys: "0", "5", "10", "last".
For each scenario, walls pick either RC or CLT capacity from that key.
"""

import math
import matplotlib.pyplot as plt

# ---------------- Colors ----------------
WALL_COLORS = ["#B8D1CA", "#5F938C", "#296872"]  # cycles for walls
CENTROID_COLORS = {
    "P₁": "#B8D1CA",
    "P₂": "#F79433",
    "P₃": "#296872",
    "P₄": "#5F938C",
}

# --------------- Geometry ---------------
walls = [
    ("W₁", "V", (1, 2), (1, 8)),   # vertical
    ("W₂", "V", (7, 3), (7, 7)),   # vertical
    ("W₃", "H", (3, 9), (7, 9)),   # horizontal (top)
    ("W₄", "H", (2, 1), (8, 1)),   # horizontal (bottom)
]

def wall_length(w):
    _, _, (x1, y1), (x2, y2) = w
    return math.hypot(x2 - x1, y2 - y1)

def wall_centroid(w):
    _, _, (x1, y1), (x2, y2) = w
    return ((x1 + x2) / 2, (y1 + y2) / 2)

centroids_wall = {w[0]: wall_centroid(w) for w in walls}

# --------------- Capacity k-tables by scenario ---------------
# Keys: scenario_key -> {"RC": {L: k}, "CLT": {L: k}}
K_TABLES = {
    "0": {   # corresponds to your "_0" data
        "RC":  {2: 204, 4: 418, 6: 631, 8: 845, 10: 1058},
        "CLT": {2: 4.39, 4: 73.72, 6: 137, 8: 205, 10: 272},
    },
    "5": {   # corresponds to your "_5" data
        "RC":  {2: 132, 4: 346, 6: 560, 8: 773, 10: 987},
        "CLT": {2: 4, 4: 74, 6: 137, 8: 205, 10: 273},
    },
    "10": {  # corresponds to your "_10" data
        "RC":  {2: 62, 4: 274, 6: 488, 8: 702, 10: 915},
        "CLT": {2: 3, 4: 74, 6: 137, 8: 205, 10: 273},
    },
    # "last" scenario – here simply taken as same as "10"
    "last": {
        "RC":  {2: 62, 4: 274, 6: 488, 8: 702, 10: 915},
        "CLT": {2: 3, 4: 74, 6: 137, 8: 205, 10: 273},
    },
}

# Order of scenarios used for P₁–P₃
SCENARIO_KEYS_FOR_MULTI = ["0", "5", "10", "last"]
# Scenario used for P₄
SCENARIO_KEY_FOR_P4 = "last"

# --------------- Material-choice per P (RC / CLT) ---------------
# These describe which material table each wall uses in a given scenario.
CENTROID_MATERIAL_CHOICE = {
    # P₁: all walls use RC capacity (for all scenarios)
    "P₁": {"W₁": "RC",  "W₂": "RC",  "W₃": "RC",  "W₄": "RC"},
    # P₂: all walls use CLT capacity (for all scenarios)
    "P₂": {"W₁": "CLT", "W₂": "CLT", "W₃": "CLT", "W₄": "CLT"},
    # P₃: hybrid – vertical walls RC, horizontal walls CLT
    "P₃": {"W₁": "RC",  "W₂": "RC",  "W₃": "CLT", "W₄": "CLT"},
    # P₄: one hybrid point at the last scenario (can be same as P₃ or different)
    "P₄": {"W₁": "RC",  "W₂": "RC",  "W₃": "CLT", "W₄": "CLT"},
}

# --------------- Loads -----------------
W_x = 1.0
W_y = 1.0
LOAD_POINT_X = 5.0
LOAD_POINT_Y = 5.0
W = math.hypot(W_x, W_y)

# --------------- Subscript helper ---------------
_SUBS_MAP = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
def subscript_digits(text: str) -> str:
    return str(text).translate(_SUBS_MAP)

# --------------- Helpers ---------------
def k_values_for_choice(material_map, scenario_key):
    """
    material_map: e.g. {"W₁":"RC", "W₂":"RC", "W₃":"CLT", "W₄":"CLT"}
    scenario_key: "0", "5", "10", or "last"
    """
    kv = {}
    table = K_TABLES[scenario_key]
    for w in walls:
        name = w[0]
        L = round(wall_length(w))
        mat = material_map[name]  # "RC" or "CLT"
        kv[name] = table[mat][L]
    return kv

def compute_centroid_custom(kv):
    # x_c from H using y, y_c from V using x; then swap (your rule)
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

def compute_results_for_choice(kv, pname_label="P"):
    x_c, y_c = compute_centroid_custom(kv)

    dx = x_c - LOAD_POINT_X
    dy = y_c - LOAD_POINT_Y
    z  = W_y * dx + W_x * dy
    M_w = z  # consistent with your earlier script

    I_vert  = sum(kv[w[0]] * (centroids_wall[w[0]][0] - x_c)**2 for w in walls if w[1] == "V")
    I_horiz = sum(kv[w[0]] * (centroids_wall[w[0]][1] - y_c)**2 for w in walls if w[1] == "H")
    I_w = I_vert + I_horiz if (I_vert + I_horiz) != 0 else 1e-12

    sum_k_vert  = sum(kv[w[0]] for w in walls if w[1] == "V") or 1e-12
    sum_k_horiz = sum(kv[w[0]] for w in walls if w[1] == "H") or 1e-12

    Rx_wind, Rx_mom, Rx_tot = {}, {}, {}
    Ry_wind, Ry_mom, Ry_tot = {}, {}, {}

    for w in walls:
        name, ori, _, _ = w
        cx_w, cy_w = centroids_wall[name]
        k = kv[name]

        if ori == "V":
            base = (k / sum_k_vert) * W_x
            mom  = (M_w / I_w) * k * (cx_w - x_c)
            Rx_wind[name] = base
            Rx_mom[name]  = mom
            Rx_tot[name]  = base + mom
        else:
            base = (k / sum_k_horiz) * W_y
            mom  = (M_w / I_w) * k * (cy_w - y_c)
            Ry_wind[name] = base
            Ry_mom[name]  = mom
            Ry_tot[name]  = base - mom

    sum_Rx_wind = sum(Rx_wind.values())
    sum_Rx_mom  = sum(Rx_mom.values())
    sum_Rx_tot  = sum(Rx_tot.values())

    sum_Ry_wind = sum(Ry_wind.values())
    sum_Ry_mom  = sum(Ry_mom.values())
    sum_Ry_tot  = sum(Ry_tot.values())

    M_from_R_wind = 0.0
    M_from_R_mom  = 0.0

    for w in walls:
        name, ori, _, _ = w
        cx_w, cy_w = centroids_wall[name]
        if ori == "V":
            lever = (cx_w - x_c)
            if name in Rx_wind: M_from_R_wind += Rx_wind[name] * lever
            if name in Rx_mom:  M_from_R_mom  += Rx_mom[name]  * lever
        else:
            lever = (cy_w - y_c)
            if name in Ry_wind: M_from_R_wind += Ry_wind[name] * lever
            if name in Ry_mom:  M_from_R_mom  += Ry_mom[name]  * lever

    M_from_R_tot = M_from_R_wind + M_from_R_mom

    # Optional print-out (can be commented out if too verbose)
    print(f"\n===== {pname_label} RESULTS =====")
    print(f"Centroid (x_c, y_c) = ({x_c:.6f}, {y_c:.6f})")
    print(f"I_vert={I_vert:.6f}, I_horiz={I_horiz:.6f}, I_w={I_w:.6f}")
    print(f"dx={dx:.6f}, dy={dy:.6f}, z={z:.6f}, M_w={M_w:.6f}")
    print(f"  ΣR_x={sum_Rx_tot:.6f} vs W_x={W_x:.6f},  ΣR_y={sum_Ry_tot:.6f} vs W_y={W_y:.6f}")
    print(f"  M_from_R_tot={M_from_R_tot:.6f} vs M_w={M_w:.6f}")

    return x_c, y_c

# --------------- Run P₁–P₄ with scenarios ---------------
centroid_points = {}  # label -> (xc, yc, base_name)

for P_name in ["P₁", "P₂", "P₃"]:
    mat_map = CENTROID_MATERIAL_CHOICE[P_name]
    for idx, scen_key in enumerate(SCENARIO_KEYS_FOR_MULTI, start=1):
        kv = k_values_for_choice(mat_map, scen_key)
        label = f"{P_name},{subscript_digits(str(idx))}"  # e.g. P₁,₁
        xc, yc = compute_results_for_choice(kv, pname_label=label)
        centroid_points[label] = (xc, yc, P_name)

# P₄: single point, last scenario only
P4_name = "P₄"
mat_map_P4 = CENTROID_MATERIAL_CHOICE[P4_name]
kv_P4 = k_values_for_choice(mat_map_P4, SCENARIO_KEY_FOR_P4)
xc4, yc4 = compute_results_for_choice(kv_P4, pname_label=P4_name)
centroid_points[P4_name] = (xc4, yc4, P4_name)

# --------------- Plot ----------------
def plot_all():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')

    ax.set_title("Rotational Centre Placement, Capacity-Based Scenarios", fontsize=14, pad=12)

    # Walls
    for idx, w in enumerate(walls):
        (name, _, (x1, y1), (x2, y2)) = w
        ax.plot([x1, x2], [y1, y2], linewidth=4,
                color=WALL_COLORS[idx % len(WALL_COLORS)])
        cx, cy = centroids_wall[name]
        ax.text(cx, cy, name, ha='center', va='center')

    # Centroid points
    for label, (xc, yc, base_name) in centroid_points.items():
        color = CENTROID_COLORS.get(base_name, "#000000")
        ax.plot(xc, yc, 'o', ms=7, color=color)
        ax.text(xc + 0.1, yc + 0.1, subscript_digits(label),
                color=color, fontsize=9)

    # Legend entries per base P (P₁–P₄)
    handles = []
    labels = []
    for pname, col in CENTROID_COLORS.items():
        handles.append(plt.Line2D([], [], marker='o', linestyle='None', color=col, markersize=7))
        labels.append(pname)
    ax.legend(handles, labels, loc='lower right', frameon=True)

    ax.grid(True)
    plt.tight_layout()
    plt.show()

plot_all()
