"""baselines 自检：验证均匀分配与注水解的正确性

检查项（对每个速度的前 n_traces 条测试轨迹）：
  1. uniform：能量精确用满、0 ≤ p ≤ p_max
  2. waterfilling（闭式）：预算不超额（相对容差 1e-10）、功率上下界
  3. KKT 最优性：内点时隙边际速率相等（"水面齐平"）、贴 p_max 时隙边际更高
  4. 与 cvxpy 独立求解对拍（速率相对差 < 1e-4）
  5. 逐轨迹 注水 ≥ 均匀（凸问题最优值不低于任意可行解）

用法::

    python -m baselines.selftest              # 默认每速度 20 条
    python -m baselines.selftest --traces 50  # 加大抽样
"""

import sys
import numpy as np

from config import SceneConfig
from channel.trace_io import load_traces
from baselines import uniform_power, waterfilling, waterfilling_cvxpy, total_rate


def run_selftest(n_traces: int = 20, verbose: bool = True) -> bool:
    cfg = SceneConfig()
    T = None  # 从 traces 读，兼容 T=200 / T=1000
    ok = True

    for speed_f in cfg.speeds_kmh:
        speed = int(speed_f)  # config 中为 float，文件名需要整数
        g_test, _, _ = load_traces(f"traces/sionna_speed{speed}_test.npz")
        T = g_test.shape[1]
        E = cfg.avg_power * T * cfg.dt_s
        rates_u, rates_w = [], []

        for i, g in enumerate(g_test[:n_traces]):
            # ---------- 1. 均匀 ----------
            p_u = uniform_power(g, cfg.avg_power, cfg.p_max)
            e_u = p_u.sum() * cfg.dt_s
            if not (abs(e_u - E) < 1e-12 and p_u.max() <= cfg.p_max + 1e-15 and (p_u >= 0).all()):
                print(f"[FAIL] uniform v={speed} #{i}: E_used={e_u:.6e} vs E={E:.6e}")
                ok = False
            r_u = total_rate(p_u, g, cfg.bandwidth_hz, cfg.noise_power, cfg.dt_s)
            rates_u.append(r_u)

            # ---------- 2/3. 注水（闭式）：约束 + KKT ----------
            p_w = waterfilling(g, cfg.avg_power, cfg.p_max, cfg.noise_power)
            e_w = p_w.sum() * cfg.dt_s
            if not (e_w <= E * (1 + 1e-10) and p_w.max() <= cfg.p_max + 1e-12 and (p_w >= 0).all()):
                print(f"[FAIL] WF 预算/边界 v={speed} #{i}: E_used={e_w:.6e} vs E={E:.6e}")
                ok = False
            # 边际速率 dR/dp = B·log2(e)·g/(N + p·g)：内点应"水面齐平"
            marg = cfg.bandwidth_hz * np.log2(np.e) * g / (cfg.noise_power + p_w * g)
            inner = (p_w > 1e-9) & (p_w < cfg.p_max - 1e-9)
            clipped = p_w >= cfg.p_max - 1e-9
            if inner.sum() >= 2:
                spread = (marg[inner].max() - marg[inner].min()) / marg[inner].mean()
                if spread > 1e-3:
                    print(f"[FAIL] WF KKT v={speed} #{i}: 内点边际散布={spread:.2e}")
                    ok = False
            if clipped.any() and inner.any() and marg[clipped].min() < marg[inner].max() - 1e-6:
                print(f"[FAIL] WF KKT 贴上界 v={speed} #{i}: 贴限时隙边际反而更低")
                ok = False
            r_w = total_rate(p_w, g, cfg.bandwidth_hz, cfg.noise_power, cfg.dt_s)
            rates_w.append(r_w)

            # ---------- 4. cvxpy 独立对拍 ----------
            p_c = waterfilling_cvxpy(g, cfg.avg_power, cfg.p_max, cfg.noise_power)
            r_c = total_rate(p_c, g, cfg.bandwidth_hz, cfg.noise_power, cfg.dt_s)
            if abs(r_w - r_c) / r_c > 1e-4:
                print(f"[FAIL] WF vs cvxpy v={speed} #{i}: 相对差={abs(r_w-r_c)/r_c:.2e}")
                ok = False

            # ---------- 5. 最优值 ≥ 可行值 ----------
            if r_w < r_u - 1e-6:
                print(f"[FAIL] 注水 < 均匀 v={speed} #{i}")
                ok = False

        if verbose:
            mu, mw = np.mean(rates_u), np.mean(rates_w)
            print(f"v={speed:>3} km/h: uniform={mu/1e6:.2f} Mb  WF={mw/1e6:.2f} Mb  "
                  f"增益={(mw/mu-1)*100:.1f}%  [T={T}, 抽查 {len(rates_u)} 条]")

    return ok


if __name__ == "__main__":
    n = 20
    if "--traces" in sys.argv:
        n = int(sys.argv[sys.argv.index("--traces") + 1])
    passed = run_selftest(n_traces=n)
    print("=== 自检全部通过 ===" if passed else "=== 存在失败项 ===")
    sys.exit(0 if passed else 1)
