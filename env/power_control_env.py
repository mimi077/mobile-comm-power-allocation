"""高速移动功率控制 Gymnasium 环境（离散动作 + 历史 CSI 版）

动作（离散 5 档，模拟 5G TPC 式功率控制）：
    0..4 → 功率档位 [0, 1/4, 1/2, 3/4, 1] × p_max
    受剩余能量硬截断（剩余能量不足时降到可支付的最高档）

观测（10 维）：
    [最近 8 个时隙的 g_dB 归一化值（含当前）, 时间进度, 剩余能量比例]
    完美当前 CSI：观测含精确当前 g_t；历史窗口提供趋势外推素材

奖励：默认"相对均匀分配的差分奖励"（关键设计）
      r_t = [rate(p_agent) - rate(p_uniform)] · dt · SCALE
      抵消不同轨迹间信道实现的巨大外生方差（可达 10 倍），
      只保留"动作比均匀分配好多少"的纯净信号；
      数学合法：Σ差分 = R_agent − R_uniform，后者对给定轨迹为常数，不改变最优策略。
      评估仍用 info["rate"] 累加的绝对速率。
"""

import os
import sys
from collections import deque

import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SceneConfig
from channel.trace_io import load_traces

REWARD_SCALE = 1e-6  # 把 bits 缩放到 ~0.1 量级，仅影响 PPO 训练，不影响评估的原始速率


class PowerControlEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, trace_path: str, cfg: SceneConfig, seed: int = 0,
                 relative_reward: bool = True):
        super().__init__()
        self.cfg = cfg
        self.relative_reward = relative_reward
        self.g_all, self.d, self.meta = load_traces(trace_path)
        self.n_traces, self.T = self.g_all.shape
        self.rng = np.random.default_rng(seed)

        # 离散功率档位 [0, 1/4, 1/2, 3/4, 1] × p_max
        self.power_levels = np.linspace(0.0, 1.0, cfg.n_power_levels) * cfg.p_max

        self.action_space = gym.spaces.Discrete(cfg.n_power_levels)
        # 观测: [g 历史(8), t/T, E_rem/E]
        obs_dim = cfg.g_history_len + 2
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([np.full(cfg.g_history_len, -2.0), [0.0, 0.0]]).astype(np.float32),
            high=np.concatenate([np.full(cfg.g_history_len, 1.0), [1.0, 1.0]]).astype(np.float32),
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.E_total = cfg.avg_power * self.T * cfg.dt_s  # 总能量预算 [J]
        self._trace_idx = 0
        self._t = 0
        self._E_rem = self.E_total
        self._g = None
        self._g_hist = deque(maxlen=cfg.g_history_len)

    def _g_to_norm(self, g: float) -> float:
        g_db = 10.0 * np.log10(max(g, 1e-30))
        return (g_db + 90.0) / 60.0  # [-120,-40] dB → [-0.5, 0.83]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # Gymnasium 规范：初始化 env 级 RNG
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._trace_idx = int(self.rng.integers(0, self.n_traces))
        self._g = self.g_all[self._trace_idx]
        self._t = 0
        self._E_rem = self.E_total
        # 历史窗口用当前 g 填充（episode 开始时尚无历史）
        self._g_hist.clear()
        g0 = self._g_to_norm(self._g[0])
        for _ in range(self.cfg.g_history_len):
            self._g_hist.append(g0)
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        return np.array(
            list(self._g_hist) + [self._t / self.T, self._E_rem / self.E_total],
            dtype=np.float32,
        )

    def step(self, action):
        level = int(action)
        p_req = float(self.power_levels[level])
        # 硬能量约束：剩余能量不够时降到可支付的最高档（保持离散档位语义）
        p_affordable = self._E_rem / self.cfg.dt_s
        if p_req > p_affordable:
            affordable_levels = self.power_levels[self.power_levels <= p_affordable + 1e-12]
            p = float(affordable_levels[-1]) if len(affordable_levels) > 0 else 0.0
        else:
            p = p_req
        self._E_rem = max(0.0, self._E_rem - p * self.cfg.dt_s)

        rate = self.cfg.bandwidth_hz * np.log2(1.0 + p * self._g[self._t] / self.cfg.noise_power)
        if self.relative_reward:
            # 均匀分配基线：每 slot 恒用 avg_power（同预算），扣除其速率以抵消信道外生方差
            p_uni = min(self.cfg.avg_power, self.cfg.p_max)
            rate_uni = self.cfg.bandwidth_hz * np.log2(
                1.0 + p_uni * self._g[self._t] / self.cfg.noise_power)
            reward = (rate - rate_uni) * self.cfg.dt_s * REWARD_SCALE
        else:
            reward = rate * self.cfg.dt_s * REWARD_SCALE

        self._t += 1
        if self._t < self.T:
            self._g_hist.append(self._g_to_norm(self._g[self._t]))
        terminated = False
        truncated = self._t >= self.T
        info = {"p": p, "rate": rate, "g": self._g[self._t - 1], "E_rem": self._E_rem}
        return self._obs(), reward, terminated, truncated, info

    def get_trace(self) -> np.ndarray:
        return self._g.copy()
