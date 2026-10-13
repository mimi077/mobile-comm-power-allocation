"""两个功率分配基线：平均分配 与 时间维注水

两者目标一致：最大化 episode 总速率 Σ B·log2(1 + p_t·g_t/N)·dt
约束一致:0 ≤ p_t ≤ p_max , Σ p_t·dt ≤ E  (即 Σ p_t ≤ avg_power·T)

区别：
  - uniform:不看信道，每时隙用相同功率（下界）
  - waterfilling:非因果，已知全程 g[1:T]，凸问题全局最优（上界）

────────────────────────────────────────────────────────────────
注水算法的原理（为什么解长这样）
────────────────────────────────────────────────────────────────
目标对 p_t 是凹的(log 的凹函数复合线性)，约束全线性 → 凸问题，
KKT 条件既必要又充分。Lagrangian 对 p_t 求导置零：

    d/dp_t [log2(1 + g_t·p_t/N)] = λ + v_t - η_t
    →  g_t / ((N + g_t·p_t)·ln2) = λ        (内点解: v_t = η_t = 0)

所有"内点"时隙的边际速率相等——这就是"水面齐平":
    p_t* = 1/λ - N/g_t ≡ μ - N/g_t

物理解释：把 N/g_t 看作时隙 t 的"地形高度"，倒水到统一水位 μ，
水只积在低洼（信道好）处：
  - 地形太高(g_t 小,N/g_t > μ)：不积水 → p_t = 0(深衰落时隙省电)
  - 水位超过容器盖(μ - N/g_t > p_max)：贴上界 → p_t = p_max
  统一写作 p_t* = clip(μ - N/g_t, 0, p_max) (μ 由预算方程 Σ p_t* = E/dt)
  唯一确定（对 μ 单调递减 → 二分求根，每次评估 O(T)）。

注意与"跨子信道注水"的差别：这里是在时间维注水（一个信道、多个时隙），
非因果性体现在 μ 依赖未来所有 g_t(全程 CSI 才能定水位)——因此它是
因果策略的性能上界，不是可在线实现的方案。

数值要点：
  - g 跨多个数量级时 log 目标条件数极差,cvxpy 直接解易失败；
    闭式二分不经过 log 目标，数值稳定且为精确解 → 作为主入口
  - 输入 g 强转 float64:float32 输入会使 p.sum() 带 ~1e-6 求和误差，
    导致解微小超出硬能量预算（曾实测 +1.0e-7 相对）
"""

import numpy as np
import cvxpy as cp


def uniform_power(g: np.ndarray, avg_power: float, p_max: float) -> np.ndarray:
    """平均分配:p_t = min(avg_power, p_max)"""
    T = len(g)
    return np.full(T, min(avg_power, p_max), dtype=np.float64)


def waterfilling_cvxpy(
    g: np.ndarray,
    avg_power: float,
    p_max: float,
    noise_power: float,
) -> np.ndarray:
    """cvxpy 时间维注水（非因果，已知全程 g）。

    max  Σ log(1 + g_t·p_t/N)
    s.t. 0 ≤ p_t ≤ p_max,  Σ p_t ≤ avg_power·T

    注意：g 跨多个数量级时 log 目标条件数极差,CLARABEL 会求解失败。
    这里对 g 做归一化（g/g_max，同步缩放 N），速率不变但数值稳定。
    """
    T = len(g)
    g = np.asarray(g, dtype=np.float64)  # 与闭式解保持一致，避免 float32 求和误差
    g0 = np.max(g)
    g_s = g / g0
    n_s = noise_power / g0
    p = cp.Variable(T, nonneg=True)
    obj = cp.Maximize(cp.sum(cp.log(1.0 + cp.multiply(g_s, p) / n_s)))
    constraints = [p <= p_max, cp.sum(p) <= avg_power * T]
    prob = cp.Problem(obj, constraints)
    # CLARABEL 在 g 跨度大时即使归一化也可能失败；SCS 对 log 目标更鲁棒
    for solver in [cp.CLARABEL, cp.SCS]:
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate"):
                break
        except cp.error.SolverError:
            continue
    if prob.status not in ("optimal", "optimal_inaccurate"):
        # 所有求解器失败则回退闭式解
        return waterfilling_closed_form(g, avg_power, p_max, noise_power)
    return np.clip(p.value, 0.0, p_max).astype(np.float64)


def waterfilling(
    g: np.ndarray,
    avg_power: float,
    p_max: float,
    noise_power: float,
) -> np.ndarray:
    """注水主入口：用闭式二分解（数值稳定、速度快、对该凸问题为精确解）。"""
    return waterfilling_closed_form(g, avg_power, p_max, noise_power)


def waterfilling_closed_form(
    g: np.ndarray,
    avg_power: float,
    p_max: float,
    noise_power: float,
) -> np.ndarray:
    """闭式二分注水（仅用于交叉验证 cvxpy 结果）。

    p_t = min(p_max, max(0, 1/μ - N/g_t))，μ 由 Σ p_t = avg_power·T 确定。
    """
    g = np.asarray(g, dtype=np.float64)  # 关键：float32 输入会使 p.sum() 有 ~1e-6 误差，导致预算微小超额
    T = len(g)
    target = avg_power * T
    lo, hi = 1e-20, 1.0 / noise_power  # μ 范围
    for _ in range(200):
        mid = (lo + hi) / 2.0
        p = np.clip(1.0 / mid - noise_power / np.maximum(g, 1e-30), 0.0, p_max)
        if p.sum() > target:
            lo = mid  # 功率超了 → 提高 mid（mid = 1/水位，等价于降低水位、减少注水）
        else:
            hi = mid
    p = np.clip(1.0 / hi - noise_power / np.maximum(g, 1e-30), 0.0, p_max)
    return p.astype(np.float64)


def total_rate(p: np.ndarray, g: np.ndarray, bandwidth: float, noise_power: float, dt: float) -> float:
    """计算总传输量 [bits]"""
    return float(np.sum(bandwidth * np.log2(1.0 + p * g / noise_power) * dt))
