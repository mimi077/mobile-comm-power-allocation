import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(7,7))

# 发射源在原点
source_x, source_y = 0, 0
ax.plot(source_x, source_y, 'ro', markersize=8, label='发射天线（全向）')

# ========== 1. 原来的球面波前（同心圆） ==========
radii = [1, 2, 3]
for r in radii:
    theta = np.linspace(0, 2*np.pi, 200)
    x = source_x + r * np.cos(theta)
    y = source_y + r * np.sin(theta)
    ax.plot(x, y, 'b--', alpha=0.6, linewidth=1.2, label='球面波前（2D截面同心圆）' if r==1 else "")

# 标注功率密度
ax.text(0, 1.1, f'球面半径 r={radii[0]}\n功率密度高', ha='center', fontsize=9)
ax.text(0, 2.1, f'球面半径 r={radii[1]}\n功率密度中等', ha='center', fontsize=9)
ax.text(0, 3.1, f'球面半径 r={radii[2]}\n功率密度低', ha='center', fontsize=9)

# ========== 2. 画建筑物（灰色矩形障碍物） ==========
# 矩形: x从2.2~3.2, y从-1.0~1.0
building_x1, building_x2 = 2.2, 3.2
building_y1, building_y2 = -1.0, 1.0
from matplotlib.patches import Rectangle
building = Rectangle((building_x1, building_y1), 
                     width=building_x2-building_x1, 
                     height=building_y2-building_y1,
                     facecolor='#888888', alpha=0.7, label='建筑物（散射体）')
ax.add_patch(building)

# ========== 3. 入射射线 + 反射射线 ==========
# 入射：从原点打到墙面的撞击点
hit_x = 2.2
hit_y = 0.6
# 入射光线
ax.plot([source_x, hit_x], [source_y, hit_y], color='orange', lw=1.5, label="入射电磁波")
# 反射光线（镜面反射示意）
reflect_end_x = hit_x - 1.6
reflect_end_y = hit_y + 1.2
ax.plot([hit_x, reflect_end_x], [hit_y, reflect_end_y], color='red', lw=1.5, label="反射散射波")

# 标记撞击点
ax.plot(hit_x, hit_y, 'ko', markersize=4)

# ========== 4. 其他径向射线 ==========
angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
for ang in angles:
    ax.plot([0, 3.3*np.cos(ang)], [0, 3.3*np.sin(ang)], color='orange', alpha=0.4, linewidth=0.8)

ax.set_xlim(-4,4)
ax.set_ylim(-4,4)
ax.set_aspect('equal')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('理想全向天线：球面波扩散 + 建筑物反射（NLOS散射示意）')
ax.legend(loc="upper right")
ax.grid(alpha=0.3)
plt.savefig('radio_scene.png', dpi=150, bbox_inches='tight')
plt.show()