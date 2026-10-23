r"""Spherical wave spreading + building reflection, physics illustration
(static sketch, not a ray-tracing computation)

Draws the physical picture behind the Friis derivation: an omnidirectional
antenna spreads power as spherical waves (density ∝ 1/r²), and signals
reflect/scatter off buildings to form NLOS multipath.

Output (same directory as this file):
    ray_tracing.png   150 dpi bitmap

Usage::

    python figures/ray_tracing.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(7, 7))

# Tx source at origin
source_x, source_y = 0, 0
ax.plot(source_x, source_y, 'ro', markersize=8, label='Tx antenna (omni)')

# ========== 1. Spherical wavefronts (concentric circles, 2D cross-section) ==========
radii = [1, 2, 3]
for r in radii:
    theta = np.linspace(0, 2 * np.pi, 200)
    x = source_x + r * np.cos(theta)
    y = source_y + r * np.sin(theta)
    ax.plot(x, y, 'b--', alpha=0.6, linewidth=1.2,
            label='spherical wavefronts' if r == 1 else "")

# Power density annotations
ax.text(0, 1.1, f'r={radii[0]}\nhigh power density', ha='center', fontsize=9)
ax.text(0, 2.1, f'r={radii[1]}\nmedium power density', ha='center', fontsize=9)
ax.text(0, 3.1, f'r={radii[2]}\nlow power density', ha='center', fontsize=9)

# ========== 2. Building (rectangular obstacle/scatterer) ==========
building_x1, building_x2 = 2.2, 3.2
building_y1, building_y2 = -1.0, 1.0
building = Rectangle((building_x1, building_y1),
                     width=building_x2 - building_x1,
                     height=building_y2 - building_y1,
                     facecolor='#888888', alpha=0.7, label='building (scatterer)')
ax.add_patch(building)

# ========== 3. Incident + reflected ray (specular: angle in = angle out) ==========
# Vertical wall at x=2.2: incident (dx, dy) reflects to (-dx, dy)
hit_x = 2.2
hit_y = 0.6
inc_dir = np.array([hit_x - source_x, hit_y - source_y])   # (2.2, 0.6)
ax.plot([source_x, hit_x], [source_y, hit_y],
        color='orange', lw=1.5, label="incident wave")
# Reflected ray: x-component flipped, y kept; length = 0.8 × incident
ref_dir = np.array([-inc_dir[0], inc_dir[1]])
ref_end = np.array([hit_x, hit_y]) + 0.8 * ref_dir
ax.plot([hit_x, ref_end[0]], [hit_y, ref_end[1]],
        color='red', lw=1.5, label="reflected/scattered wave")

# Hit point marker
ax.plot(hit_x, hit_y, 'ko', markersize=4)

# ========== 4. Other radial rays ==========
angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
for ang in angles:
    ax.plot([0, 3.3 * np.cos(ang)], [0, 3.3 * np.sin(ang)],
            color='orange', alpha=0.4, linewidth=0.8)

ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Omnidirectional antenna: spherical wave spreading + building reflection')
ax.legend(loc="upper right")
ax.grid(alpha=0.3)
# 保存到本脚本所在目录（与运行时 cwd 无关）
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ray_tracing.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"saved: {out}")
