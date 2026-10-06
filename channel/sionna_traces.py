"""阶段B：Sionna TDL (TR 38.901) 小尺度衰落 + 几何大尺度 → 每时隙信道增益轨迹

架构约定（解耦）：
    信道建模只在离线阶段运行，产物为 traces/*.npz；
    RL 环境/基线/评估只读 npz，绝不 import sionna。

模型构成：
    小尺度：TDL-C (NLOS) 抽头系数相干合成 → g_small(t)=|Σ_p a_p(t)|²，E=1，
            时间相关性由 max_doppler = v·f_c/c 决定（Jakes 谱）
    大尺度：自由空间路径损耗 PL(d_t)（U形几何）× 每轨迹恒定对数正态阴影
    总增益：g_t = 10^{-(PL+shadow)/10} · g_small(t)

用法：
    python channel/sionna_traces.py --quick   # 小规模快速验证（20+6 条）
    python channel/sionna_traces.py           # 全量（1000+100 × 3 速度）
"""

import argparse
import os
import sys
import time
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SceneConfig

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.special import j0
from sionna.phy import config as sionna_config
from sionna.phy.channel.tr38901 import TDL

import sionna

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRACE_DIR = os.path.join(ROOT, "traces")
RESULT_DIR = os.path.join(ROOT, "results")

CHUNK = 100  # 每批生成轨迹数（控制 TDL 中间张量内存）


def generate_speed(cfg: SceneConfig, speed_kmh: float, n_traces: int, seed: int):
    """生成一个速度的 n_traces 条轨迹。

    返回:
        g     [N, T] float32 总功率增益（线性）
        d     [T]    float32 距离
        stats dict   小尺度统计（均值功率、lag-1 相关、经验 ACF）
    """
    d = cfg.distance_m(speed_kmh)                # [T]
    g_ls_time = 10.0 ** (-cfg.fspl_db(d) / 10.0)  # [T] 无阴影大尺度
    f_d = cfg.f_doppler(speed_kmh)

    # Sionna：固定速度 → min_speed = max_speed（内部按载频换算多普勒）
    sionna_config.seed = seed
    sionna_config.device = "cpu"
    tdl = TDL(
        model=cfg.tdl_profile,
        delay_spread=cfg.tdl_delay_spread_s,
        carrier_frequency=cfg.f_c_hz,
        num_sinusoids=cfg.num_sinusoids,
        min_speed=cfg.speed_to_mps(speed_kmh),
        max_speed=cfg.speed_to_mps(speed_kmh),
        device="cpu",
    )

    rng = np.random.default_rng(seed + 1)  # 阴影衰落独立于 TDL 随机流
    g = np.empty((n_traces, cfg.T), np.float32)
    g_small_first = None  # 留一批用于统计
    done = 0
    while done < n_traces:
        b = min(CHUNK, n_traces - done)
        a, _ = tdl(batch_size=b, num_time_steps=cfg.T,
                   sampling_frequency=1.0 / cfg.dt_s)   # 每时隙采样一次
        h = a.sum(dim=5).reshape(b, cfg.T)              # 窄带合成 → [b, T]
        g_small = (h.abs() ** 2).numpy().astype(np.float32)
        if g_small_first is None:
            g_small_first = g_small.copy()
        shadow_lin = 10.0 ** (rng.normal(0.0, cfg.shadow_sigma_db, size=(b, 1)) / 10.0)
        g[done:done + b] = g_ls_time[None, :] * shadow_lin * g_small
        done += b

    # ---- 小尺度统计（相关性是速度对比实验的命脉，生成时即验证）----
    # 注意：功率增益 |h|² 的相关理论值为 J0²（Rayleigh 过程），不是 J0
    m = float(g_small_first.mean())
    lags = np.arange(1, 21)
    acf = []
    x0 = g_small_first - g_small_first.mean(axis=1, keepdims=True)
    for lag in lags:
        x, y = x0[:, :-lag], x0[:, lag:]
        num = (x * y).sum(axis=1)
        den = np.sqrt((x ** 2).sum(axis=1) * (y ** 2).sum(axis=1))
        acf.append(float((num / np.maximum(den, 1e-12)).mean()))
    stats = {
        "f_doppler_hz": f_d,
        "mean_power": m,
        "lag1_empirical": acf[0],
        "lag1_theory_j02": float(j0(2.0 * np.pi * f_d * cfg.dt_s) ** 2),
        "acf_lags": lags.tolist(),
        "acf_empirical": acf,
        "acf_theory": [float(j0(2.0 * np.pi * f_d * lag * cfg.dt_s) ** 2)
                       for lag in lags],
    }
    return g, d, stats


def plot_sanity(all_stats, quick: bool) -> None:
    # 注意图内文字用英文（matplotlib 默认字体无 CJK 字形，避免乱码方框）
    tag = "_quick" if quick else ""
    # 图1：三个速度各一条测试轨迹的总增益时序（dB）
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for ax, (v, stats) in zip(axes, all_stats.items()):
        z = np.load(os.path.join(TRACE_DIR, f"sionna_speed{int(v)}_test.npz"))
        g_db = 10.0 * np.log10(z["g"][0])
        t = np.arange(len(g_db))
        ax.plot(t, g_db, lw=0.6)
        ax.set_ylabel("g [dB]")
        ax.set_title(f"v = {int(v)} km/h  (f_D = {stats['f_doppler_hz']:.0f} Hz, "
                     f"lag-1 rho = {stats['lag1_empirical']:+.2f})")
    axes[-1].set_xlabel("slot t [ms]")
    fig.suptitle("Sionna TDL-C channel gain traces (test trace #1 per speed)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, f"sionna_sanity_timeseries{tag}.png"), dpi=130)
    plt.close(fig)

    # 图2：经验 ACF vs J0² 理论曲线（功率增益相关 = J0²）
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tab:blue", "tab:orange", "tab:red"]
    for c, (v, stats) in zip(colors, all_stats.items()):
        ax.plot(stats["acf_lags"], stats["acf_empirical"], "o-", color=c,
                label=f"{int(v)} km/h empirical")
        ax.plot(stats["acf_lags"], stats["acf_theory"], "--", color=c, alpha=0.6,
                label=f"{int(v)} km/h J0² theory")
    ax.set_xlabel("lag [slots]")
    ax.set_ylabel("normalized autocorrelation")
    ax.set_title("Small-scale gain temporal ACF: empirical vs J0² theory")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, f"sionna_sanity_acf{tag}.png"), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="小规模快速验证")
    args = ap.parse_args()

    cfg = SceneConfig()
    if args.quick:
        cfg = replace(cfg, n_train=20, n_test=6)

    os.makedirs(TRACE_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    all_stats = {}

    for i, v in enumerate(cfg.speeds_kmh):
        for split, n in (("train", cfg.n_train), ("test", cfg.n_test)):
            # 种子按 (速度, split) 隔离：测试集三组方法共享、可复现
            seed = cfg.seed + 10_000 * i + (0 if split == "train" else 5_000_000)
            t0 = time.time()
            g, d, stats = generate_speed(cfg, v, n, seed)
            f_d = stats["f_doppler_hz"]
            meta = {
                "source": "sionna_tdl",
                "sionna_version": sionna.__version__,
                "torch_version": torch.__version__,
                "speed_kmh": v, "f_doppler_hz": f_d,
                "tdl_profile": cfg.tdl_profile,
                "delay_spread_s": cfg.tdl_delay_spread_s,
                "num_sinusoids": cfg.num_sinusoids,
                "f_c_hz": cfg.f_c_hz, "dt_s": cfg.dt_s, "T": cfg.T,
                "d_perp_m": cfg.d_perp_m,
                "shadow_sigma_db": cfg.shadow_sigma_db,
                "split": split, "n_traces": n, "seed": seed,
                "g_shape": "[N,T] 线性功率增益(含路径损耗+阴影+小尺度)",
                "d_shape": "[T] 车辆-节点距离[m]，同速度共享",
            }
            path = os.path.join(TRACE_DIR, f"sionna_speed{int(v)}_{split}.npz")
            from trace_io import save_traces
            save_traces(path, g, d, meta)
            print(f"v={int(v):3d} km/h {split:5s} n={n:4d}  f_D={f_d:6.1f} Hz  "
                  f"E[g_small]={stats['mean_power']:.3f}  "
                  f"lag1 实测={stats['lag1_empirical']:+.3f}  "
                  f"J0² 理论={stats['lag1_theory_j02']:+.3f}  "
                  f"({time.time() - t0:.1f}s)")
            if split == "test":
                all_stats[v] = stats

    plot_sanity(all_stats, args.quick)
    print(f"完成。轨迹 → {TRACE_DIR}/\nsanity 图 → {RESULT_DIR}/")


if __name__ == "__main__":
    main()
