"""baselines 包：平均分配 + 注水算法（时间维功率分配的两个对照基线）

公开 API（统一从这里导入）::

    from baselines import uniform_power, waterfilling, total_rate

两个基线解决同一个优化问题::

    max  Σ_t B·log2(1 + p_t·g_t/N) · dt
    s.t. 0 ≤ p_t ≤ p_max,  Σ_t p_t·dt ≤ E = avg_power·T·dt

区别仅在信息条件：
  - uniform_power : 不看信道，p_t = avg_power（恒定），下界
  - waterfilling  : 非因果，已知全程 g[0:T]，凸问题全局最优，上界

自检（验证预算/约束/KKT/与 cvxpy 对拍）::

    python -m baselines.selftest
"""

from .power_alloc import (
    uniform_power,
    waterfilling,
    waterfilling_closed_form,
    waterfilling_cvxpy,
    total_rate,
)

__all__ = [
    "uniform_power",
    "waterfilling",
    "waterfilling_closed_form",
    "waterfilling_cvxpy",
    "total_rate",
]
