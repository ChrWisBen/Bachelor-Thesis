# -*- coding: utf-8 -*-
"""
Four centroid variants on one plot.
Now renamed:
    C₁ → P₁
    C₂ → P₂
    C₃ → P₃
    C₄ → P₄
+ Prints wind (base) vs moment reaction components separately.
"""

import math
import matplotlib.pyplot as plt

# ---------------- Colors ----------------
WALL_COLORS = ["#B8D1CA", "#5F938C", "#296872"]  # cycles for walls
CENTROID_COLORS = {
    "P₁": "#F79433",
    "P₂": "#B8D1CA",
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
    _, _, (x1,y1), (x2,y2) = w
    return math.hypot(x2 - x1, y2 - y1)

def wall_centroid(w):
    _, _, (x1,y1), (x2,y2) = w
    return ((x1 + x2)/2, (y1 + y2)/2)

centroids_wall = {w[0]: wall_centroid(w) for w in walls}

# --------------- FULL k TABLE ---------------
K_TABLES = {
    "k_capacity_N0": {2:96000, 4:192000, 6:288000, 8:384000, 10:480000}, #Shear CLT  

    "k_capacity_N200": {2:1548000, 4:3096000, 6:4644000, 8:6192000, 10:7740000}, # Shear RC

    "k_unit": {2:82500, 4:660000, 6:2227500, 8:5280000, 10:10312500}, # Bending CLT 
    

    "k_shear": {2:225000, 4:1800000, 6:6075000, 8:14400000, 10:28125000}, # Bending RC

    "k_bend": {
        2: 82500,
        4: 660000,
        6: 2227500,
        8: 5280000,
        10: 10312500,
    },
}


# --------------- Define k-metric per centroid (RENAMED) ---------------
CENTROID_K_CHOICE = {
    "P₁": {"W₁":"k_capacity_N0", "W₂":"k_capacity_N200", "W₃":"k_capacity_N200", "W₄":"k_capacity_N0"},
    "P₂": {"W₁":"k_unit",          "W₂":"k_shear",          "W₃":"k_shear",        "W₄":"k_unit"},
    "P₃": {"W₁":"k_shear",         "W₂":"k_shear",         "W₃":"k_shear",       "W₄":"k_shear"},
    "P₄": {"W₁":"k_bend",          "W₂":"k_bend",          "W₃":"k_bend",        "W₄":"k_bend"},
}

# --------------- Loads -----------------
W_x = 1.0
W_y = 1.0
LOAD_POINT_X = 5.0
LOAD_POINT_Y = 5.0
W = math.hypot(W_x, W_y)

# --------------- Helpers ---------------
def k_values_for_choice(choice_map):
    kv = {}
    for w in walls:
        name = w[0]
        L = round(wall_length(w))
        metric = choice_map[name]
        kv[name] = K_TABLES[metric][L]
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

def compute_results_for_choice(kv, pname="P"):
    x_c, y_c = compute_centroid_custom(kv)

    # Wind lever and "torsional" moment (using your current definition M_w = z)
    dx = x_c - LOAD_POINT_X
    dy = y_c - LOAD_POINT_Y
    z  = W_y * dx + W_x * dy
    M_w = z  # NOTE: matches your latest script (not |W|*z)

    # Inertia consistent with reaction levers
    I_vert  = sum(kv[w[0]] * (centroids_wall[w[0]][0] - x_c)**2 for w in walls if w[1] == "V")
    I_horiz = sum(kv[w[0]] * (centroids_wall[w[0]][1] - y_c)**2 for w in walls if w[1] == "H")
    I_w = I_vert + I_horiz if (I_vert + I_horiz) != 0 else 1e-12

    sum_k_vert  = sum(kv[w[0]] for w in walls if w[1] == "V") or 1e-12
    sum_k_horiz = sum(kv[w[0]] for w in walls if w[1] == "H") or 1e-12

    # Split reactions into wind/base and moment components
    Rx_wind, Rx_mom, Rx_tot = {}, {}, {}
    Ry_wind, Ry_mom, Ry_tot = {}, {}, {}

    for w in walls:
        name, ori, _, _ = w
        cx, cy = centroids_wall[name]
        k = kv[name]

        if ori == "V":
            base = (k / sum_k_vert) * W_x
            mom  = (M_w / I_w) * k * (cx - x_c)
            Rx_wind[name] = base
            Rx_mom[name]  = mom
            Rx_tot[name]  = base + mom
        else:
            base = (k / sum_k_horiz) * W_y
            mom  = (M_w / I_w) * k * (cy - y_c)
            Ry_wind[name] = base
            Ry_mom[name]  = mom
            Ry_tot[name]  = base - mom

    # Force sums
    sum_Rx_wind = sum(Rx_wind.values())
    sum_Rx_mom  = sum(Rx_mom.values())
    sum_Rx_tot  = sum(Rx_tot.values())

    sum_Ry_wind = sum(Ry_wind.values())
    sum_Ry_mom  = sum(Ry_mom.values())
    sum_Ry_tot  = sum(Ry_tot.values())

    # Moments of each component about (x_c, y_c)
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

    # Totals
    M_from_R_tot = M_from_R_wind + M_from_R_mom

    # ---------- PRINT BLOCK ----------
    print(f"\n===== {pname} RESULTS =====")
    print(f"Centroid (x_c, y_c) = ({x_c:.6f}, {y_c:.6f})")
    print(f"I_vert={I_vert:.6f}, I_horiz={I_horiz:.6f}, I_w={I_w:.6f}")
    print(f"dx={dx:.6f}, dy={dy:.6f}, z={z:.6f}, M_w={M_w:.6f}")

    print("\n-- Vertical walls (R_x) --")
    for name in [w[0] for w in walls if w[1] == "V"]:
        print(f"  {name}: wind={Rx_wind[name]:.6f}  moment={Rx_mom[name]:.6f}  total={Rx_tot[name]:.6f}")
    print(f"  Σ wind R_x = {sum_Rx_wind:.6f} | Σ moment R_x = {sum_Rx_mom:.6f} | Σ total R_x = {sum_Rx_tot:.6f} (vs W_x={W_x:.6f})")

    print("\n-- Horizontal walls (R_y) --")
    for name in [w[0] for w in walls if w[1] == "H"]:
        print(f"  {name}: wind={Ry_wind[name]:.6f}  moment={Ry_mom[name]:.6f}  total={Ry_tot[name]:.6f}")
    print(f"  Σ wind R_y = {sum_Ry_wind:.6f} | Σ moment R_y = {sum_Ry_mom:.6f} | Σ total R_y = {sum_Ry_tot:.6f} (vs W_y={W_y:.6f})")

    print("\n-- Moments about (x_c, y_c) --")
    print(f"  From wind parts  = {M_from_R_wind:.6f}")
    print(f"  From moment parts= {M_from_R_mom:.6f}")
    print(f"  Total from Rx/Ry = {M_from_R_tot:.6f}  (vs M_w={M_w:.6f})")
    print(f"  Force checks:  ΣR_x={sum_Rx_tot:.6f} vs {W_x:.6f},  ΣR_y={sum_Ry_tot:.6f} vs {W_y:.6f}")

    return x_c, y_c

# --------------- Run all four (P₁–P₄) ---------------
centroid_points = {}
for pname, choice in CENTROID_K_CHOICE.items():
    kv = k_values_for_choice(choice)
    centroid_points[pname] = compute_results_for_choice(kv, pname=pname)

# --------------- Plot ----------------
def plot_all():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect('equal')

    # ---- ADD TITLE HERE ----
    ax.set_title("Rotational Centre Placement H=4, Wx,Wy = 1", fontsize=14, pad=12)
    # You can edit freely:
    # ax.set_title("My Custom Title", fontsize=14, pad=12)

    # Walls
    for idx, w in enumerate(walls):
        (name, _, (x1, y1), (x2, y2)) = w
        ax.plot([x1, x2], [y1, y2], linewidth=4,
                color=WALL_COLORS[idx % len(WALL_COLORS)])
        cx, cy = centroids_wall[name]
        ax.text(cx, cy, name, ha='center', va='center')

    # Centroid points
    for pname, (xc, yc) in centroid_points.items():
        ax.plot(xc, yc, 'o', ms=9, color=CENTROID_COLORS[pname], label=pname)

    ax.legend(loc='lower right', frameon=True)
    ax.grid(True)
    plt.tight_layout()
    plt.show()

plot_all()