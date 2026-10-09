r"""绘制 pass-by 场景几何图：直线路俯视 + U 形距离曲线

输出（与本文件同目录）：
    scene_geometry.pdf   矢量图，供 LaTeX 论文插入（\includegraphics）
    scene_geometry.png   150 dpi 位图，供 notebook / 快速预览

用法::

    python figures/plot_scene.py

所有几何数值（d_perp、速度、T、dt、端点位置）均从 config.SceneConfig 读取，
修改参数后重跑本脚本即可，不要在图上手改数字。
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无显示环境下也能出图
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# 允许从项目根目录直接运行：python figures/plot_scene.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SceneConfig


def plot_scene(out_dir: str) -> None:
    cfg = SceneConfig()

    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.22)

    # ================= 左图：俯视几何 =================
    ax = fig.add_subplot(gs[0, 0])

    # 直线路（车真实的运动轨迹）
    ax.axhline(0, color="#444", lw=9, solid_capstyle="round", alpha=0.85, zorder=1)
    ax.text(-68, 5.2, "Straight road", fontsize=11, color="#444", style="italic")
    ax.annotate("", xy=(72, 0), xytext=(48, 0),
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color="crimson"), zorder=3)
    ax.text(57, 4.5, "velocity $v$", color="crimson", fontsize=12)

    # 以接收机为圆心的等距环（蓝色虚线）
    th = np.linspace(0, 2 * np.pi, 400)
    for r in (cfg.d_perp_m, 35.0, 52.5):
        ax.plot(r * np.cos(th), -cfg.d_perp_m + r * np.sin(th),
                color="#1f77b4", lw=0.9, ls="--", alpha=0.35, zorder=0)
    ax.text(-73, 16.5,
            "blue dashed: range rings (constant $d$)\n"
            "orange dotted: uplink wavefronts\n"
            "(concentric: $f_D/f_c\\approx3\\times10^{-7}$ is negligible)",
            fontsize=8.5, color="#333",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", alpha=0.9))

    # 波前：以最近点处的车为圆心的同心圆。
    # 注：载波多普勒 f_D/f_c ~ 3e-7，波前压缩不可观测，故同心圆是物理准确的。
    # 多普勒的真正作用是信道时变率 f_D*dt（控制时隙相关性），见右图说明。
    for r in (9, 17, 25, 33, 41):
        xx, yy = r * np.cos(th), r * np.sin(th)
        m = yy < -3
        ax.plot(xx[m], yy[m], color="#ff7f0e", lw=0.9, ls=":", alpha=0.55, zorder=0)

    # 静止接收机
    rx, ry = 0.0, -cfg.d_perp_m
    ax.plot(rx, ry, marker="^", ms=17, color="#d62728", zorder=5)
    ax.text(3, ry - 4.5, "Receiver (static)", fontsize=11, color="#d62728")
    ax.plot([0, 0], [0, -cfg.d_perp_m], ls=":", color="green", lw=2)
    ax.text(2, -cfg.d_perp_m / 2, r"$d_{\perp}=%g$ m" % cfg.d_perp_m,
            color="green", fontsize=11)

    # 车的三个位置：端点取 350 km/h episode 边界，中点为最近点
    x_end = cfg.speed_to_mps(350.0) * (cfg.T - 1) / 2.0 * cfg.dt_s  # ≈48.6 m
    car_xs = [-x_end, 0.0, x_end]
    labels = [r"approaching $t<T/2$", r"closest point $t=T/2$", r"leaving $t>T/2$"]
    offsets = [(-2, -14.5), (0, 11), (-2, 11)]
    colors = ["#8c564b", "#2ca02c", "#8c564b"]
    for x, lab, (dx, dy), c in zip(car_xs, labels, offsets, colors):
        ax.plot([x, rx], [0, ry], ls="--", lw=1.1, color=c, alpha=0.75, zorder=2)
        ax.add_patch(FancyBboxPatch((x - 4.2, -2.6), 8.4, 5.2,
                                    boxstyle="round,pad=0,rounding_size=1.2",
                                    fc="#ffbf00", ec="black", lw=1.2, zorder=4))
        ax.plot([x - 2.8, x + 2.8], [-2.6, -2.6], "o", ms=3.2, color="black", zorder=5)
        ax.text(x + dx, dy, lab, fontsize=9.5, ha="center", color=c)

    ax.text(-30, -40,
            r"$d(t)=\sqrt{d_{\perp}^2 + x(t)^2}$,  $x(t)=v\,(t-T/2)$",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#999"))
    ax.set_xlim(-75, 80)
    ax.set_ylim(-55, 32)
    ax.set_aspect("equal")
    ax.set_title("Top view: the ROAD is a straight line (the 'U' is not the road!)",
                 fontsize=11.5)
    ax.axis("off")

    # ================= 右图：d(t) 的 U 形曲线 =================
    ax2 = fig.add_subplot(gs[0, 1])
    t = np.arange(cfg.T) * cfg.dt_s
    for v, m in zip(cfg.speeds_kmh, ("o", "s", "^")):
        ax2.plot(t, cfg.distance_m(v), marker=m, ms=3, markevery=150,
                 lw=1.6, label=f"{int(v)} km/h")
    ax2.axvline(cfg.episode_s() / 2, color="grey", ls=":", lw=1.2)
    ax2.set_xlabel("time within episode $t$ [s]")
    ax2.set_ylabel("vehicle-receiver distance $d(t)$ [m]")
    ax2.set_title(r"$d(t)$ is U-shaped (symmetric), not the road", fontsize=11.5)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    # 多普勒的真正作用：不是载波偏移(f_D/f_c~3e-7)，而是信道时变率 f_D*dt
    fd_note = "\n".join([r"Doppler's role: $f_D\Delta t$ sets slot correlation",
                         r"$f_D=vf_c/c$;  $T_c\sim 1/(2f_D)$  (coherence time)"])
    ax2.text(0.02, 0.97, fd_note, transform=ax2.transAxes,
             fontsize=7.5, color="#555", va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ddd", alpha=0.85))

    fig.suptitle(r"Pass-by episode geometry: straight road + static receiver "
                 r"offset by $d_{\perp}$", fontsize=13, y=0.99)

    pdf_path = os.path.join(out_dir, "scene_geometry.pdf")
    png_path = os.path.join(out_dir, "scene_geometry.png")
    fig.savefig(pdf_path, bbox_inches="tight")          # 矢量 PDF（LaTeX 用）
    fig.savefig(png_path, dpi=150, bbox_inches="tight")  # 位图预览
    plt.close(fig)
    print(f"saved: {pdf_path}")
    print(f"saved: {png_path}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    plot_scene(here)
