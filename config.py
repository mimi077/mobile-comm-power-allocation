"""高速移动功率控制实验 —— 场景与仿真参数

信道生成(阶段A解析/阶段B Sionna)、Gym环境、基线、评估共用本配置。
"""

import numpy as np
from dataclasses import dataclass

C0 = 299_792_458.0  # 光速 [m/s]


@dataclass(frozen=True) 
class SceneConfig:
    # ---- 物理参数 ----
    f_c_hz: float = 2.0e9            # 载波频率 [Hz]

    # ---- 几何：直线道路，静止接收节点位于横向偏移 d_perp 处 ----
    d_perp_m: float = 20.0           # 道路到节点的垂直距离 [m]

    # ---- 时隙结构 ----
    dt_s: float = 1.0e-3             # 决策时隙 1 ms（保证低速时隙间信道强相关）
    T: int = 1000                    # episode 时隙数（共 1 s，最近点在 episode 中点）

    # ---- 速度场景 [km/h]：低 / 中 / 高 ----
    speeds_kmh: tuple = (30.0, 120.0, 350.0)

    # ---- 大尺度 ----
    shadow_sigma_db: float = 4.0     # 阴影衰落 std [dB]（每条轨迹内恒定）

    # ---- 功率 / 速率 ----
    # 选 noise=1e-8, avg=0.05, p_max=0.2：中位 SNR≈-2dB（信道受限区），
    # 注水相对均匀分配增益约 +20%，三组对比有明显区分度。
    bandwidth_hz: float = 10.0e6     # 带宽 [Hz]
    noise_power: float = 1.0e-8      # 等效接收噪声功率 [W]
    p_max: float = 0.2               # 单时隙最大发射功率 [W] 峰值约束
    avg_power: float = 0.05          # 平均功率约束 [W]
    n_power_levels: int = 5          # 离散功率档位数（含 0 档），对应 5G TPC 式离散功率控制
    g_history_len: int = 8           # 观测包含当前完美CSI + 最近7步历史增益（趋势素材窗口）


    # ---- 小尺度（Sionna TDL, TR 38.901）----
    tdl_profile: str = "C"           # A-E 可伸缩 profile；C = NLOS 富散射
    tdl_delay_spread_s: float = 300.0e-9   # TDL-C 名义 RMS 时延扩展 [s]
    num_sinusoids: int = 20          # TDL SoS 多普勒仿真正弦波数

    # ---- 数据规模 ----
    n_train: int = 1000              # 每速度训练轨迹数
    n_test: int = 100                # 每速度测试轨迹数（固定种子，三组方法共享）
    seed: int = 2026                 # 基础随机种子

    # ================= 派生量 =================
    def speed_to_mps(self, kmh: float) -> float:
        return kmh / 3.6

    def f_doppler(self, speed_kmh: float) -> float:
        """最大多普勒频移 [Hz]：f_D = v·f_c/c"""
        return self.speed_to_mps(speed_kmh) * self.f_c_hz / C0

    def episode_s(self) -> float:
        return self.T * self.dt_s

    def distance_m(self, speed_kmh: float) -> np.ndarray:
        """episode 内各时隙的车-节点距离 [T]（最近点位于 episode 中点）"""
        v = self.speed_to_mps(speed_kmh)
        t = np.arange(self.T) * self.dt_s
        x = v * (t - (self.T - 1) * self.dt_s / 2.0)
        return np.sqrt(self.d_perp_m ** 2 + x ** 2)

    def fspl_db(self, d_m) -> np.ndarray:
        """自由空间路径损耗 [dB]"""
        return 20.0 * np.log10(4.0 * np.pi * np.asarray(d_m) * self.f_c_hz / C0)
