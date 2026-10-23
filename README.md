# mobile-comm-power-allocation

高速移动车辆 ↔ 静止接收节点的**时隙功率分配**对比实验：平均分配 vs 注水（非因果上界），
在低/中/高三个速度场景下比较。信道由 Sionna TDL-C 离线生成，Gymnasium 环境已就绪；
**学习策略与评估代码尚未入库**（见 §5 状态说明）。

## 1. 问题设定

车沿直路经过一个静止接收节点（最近点在 episode 中点），每时隙决策一次发射功率：

- 目标：最大化 episode 总传输量 $\sum_t B\log_2(1+p_t g_t / N_0)\Delta t$
- 约束：硬能量预算 $\sum_t p_t\Delta t \le E$，$0\le p_t\le p_{\max}$
- 信道 $g_t$：外生序列（功率不影响信道），**离线预生成**，训练/评估时查表

| 对比组 | 方法 | 信息条件 | 实现状态 |
|---|---|---|---|
| 平均分配 | $p_t=\bar P$ 恒定 | 无信道信息（下界） | ✓ 已实现并自检 |
| 注水 | $p_t^*=\mathrm{clip}(\mu - N/g_t,\,0,\,p_{\max})$ | 非因果，全程 CSI（理论上界） | ✓ 已实现并自检 |
| 学习策略 | 离散 5 档 TPC 式决策 | 因果（当前/历史 CSI + 剩余能量） | 环境 ✓，策略与评估 ✗ 未入库 |

速度场景：**30 / 120 / 350 km/h**，$\Delta t=1$ ms，$T=1000$（episode 1 s）。
两个基线读同一批测试轨迹（每速度 100 条，固定种子），严格可比。

## 2. 目录结构与当前状态

```text
mobile-comm-power-allocation/
├── README.md                        ← 本文件
├── requirements.txt                 依赖冻结（rl-pc 环境）
├── config.py                        场景参数唯一来源（所有模块共用）
├── config.ipynb                     参数选择依据讲解（含 LaTeX）
├── channel/
│   ├── sionna_traces.py      [✓]    Sionna TDL-C + FSPL + 每轨迹恒定 4 dB 阴影
│   ├── sionna_traces.ipynb          信道建模讲解版
│   ├── trace_io.py           [✓]    npz 读写（数据契约，下游唯一入口）
│   └── trace_io.ipynb               数据格式讲解
├── traces/                   [✓]    6 个 npz：3 速度 × {train 1000×1000, test 100×1000}
├── baselines/
│   ├── power_alloc.py        [✓]    平均分配 + 注水（闭式二分主解，cvxpy 对拍）
│   ├── power_alloc.ipynb            API 黑箱指南 + 逐函数走读
│   ├── selftest.py           [✓]    5 项正确性检查（自动出图）
│   └── selftest.ipynb               5 项检查的原理讲解
├── env/power_control_env.py  [✓]    Gymnasium 环境（离散 5 档 + 8 步历史 + 差分奖励）
├── figures/
│   ├── plot_scene.py                → scene_geometry.pdf / .png
│   ├── ray_tracing.py               → ray_tracing.png（传播示意图）
│   └── selftest_baselines.png       由 selftest 生成
└── results/                  [本地生成，未入库]  信道 sanity 图
```

**数据规模**：traces 与 `config.py` 的 `T=1000`、`n_train=1000`、`n_test=100` 一致，
无需再重生成。`results/` 被 `.gitignore` 忽略，克隆后首次跑信道生成才会出现。

## 3. 信道数据（离线生成，下游不依赖 Sionna）

### 3.1 模型构成

| 分量 | 实现 | 说明 |
|---|---|---|
| 小尺度 | Sionna `TDL`（TR 38.901 TDL-C，NLOS） | 20 正弦波 SoS 多普勒；$f_D=v f_c/c$ |
| 路径损耗 | `fspl_db()` + U 形几何 | 车-节点直线距离，最近点在中点 |
| 阴影 | 对数正态，$\sigma=4$ dB | **每轨迹恒定**，不随时隙变化 |

窄带合成：$g_{\text{small}}(t)=|\sum_p a_p(t)|^2$，再乘大尺度得 $g_t$。

**小尺度分量 lag-1 相关实测 0.94 / 0.32 / 0.15**，与 $J_0^2(2\pi f_D\Delta t)$ 理论值
0.94 / 0.32 / 0.15 一致（这是 $|h|^2$ 的相关，不是 $h$ 的 $J_0$——本阶段最易错处）。
含路损的总增益 lag-1 为 0.94 / 0.33 / 0.32：350 km/h 被 U 形路损弧抬高，
不要用总增益去验证 Jakes 理论。

### 3.2 数据契约（npz）

| 键 | 形状 / 类型 | 含义 |
|---|---|---|
| `g` | `[N, T]` float32 | 线性功率增益，已含路径损耗 + 阴影 + 小尺度 |
| `d` | `[T]` float32 | 各时隙车-节点距离 [m]，同速度所有轨迹共享 |
| `meta` | JSON 字符串 | 场景元数据：sionna/torch 版本、速度、$f_D$、TDL profile、$\Delta t$、种子等 |

读写统一走 `channel/trace_io.py`；`env`、`baselines` 只读 npz，绝不 import sionna。

### 3.3 生成

```bash
python channel/sionna_traces.py --quick   # 小规模快验：20 train + 6 test
python channel/sionna_traces.py           # 全量：1000 + 100 × 3 速度（CPU）
```

种子按 (速度, split) 隔离；测试集固定种子，两个基线与后续策略共享同一批轨迹。
生成时打印 `lag1 实测 ≈ J0² 理论`，并把 ACF sanity 图写入 `results/`。

## 4. 基线（baselines 包，已验证）

### 4.1 使用

```python
from baselines import uniform_power, waterfilling, total_rate

p_uni = uniform_power(g, avg_power, p_max)          # 恒定功率，下界
p_wf  = waterfilling(g, avg_power, p_max, noise)    # 闭式二分注水，非因果上界
bits  = total_rate(p_wf, g, bandwidth, noise, dt)   # 总传输量 [bit]
```

主入口 `waterfilling` = 闭式二分解（数值稳定、对该凸问题精确）；
`waterfilling_cvxpy` 保留用于独立对拍。**原理与 KKT 推导见
[baselines/power_alloc.py](baselines/power_alloc.py) 模块文档**。

### 4.2 自检（一条命令回归验证）

```bash
python -m baselines.selftest            # 每速度抽查 20 条测试轨迹
python -m baselines.selftest --traces 50
```

检查项：① 能量预算不超额（含相对容差 1e-10）；② 功率上下界；
③ KKT——内点时隙边际速率相等（"水面齐平"）、贴 p_max 时隙边际更高；
④ 与 cvxpy 独立求解对拍（速率相对差 < 1e-4）；⑤ 逐轨迹注水 ≥ 均匀。
运行后自动更新 `figures/selftest_baselines.png`（速率对比柱状图 + 示例轨迹功率分配）。

### 4.3 参考数字（全量 100 条测试轨迹，T=1000）

| 速度 | 均匀 | 注水 | 注水增益 |
|---|---|---|---|
| 30 km/h | 14.01 Mb | 15.39 Mb | +9.9% |
| 120 km/h | 12.77 Mb | 14.24 Mb | +11.5% |
| 350 km/h | 8.56 Mb | 10.35 Mb | **+21.0%** |

**增益随速度上升**：U 形路损 + 每轨迹恒定阴影提供了所有速度共有的大动态范围，
而速度越高小尺度衰落去相关越彻底，时隙间起伏更剧烈——注水可搬移的能量差更大。
证据是深衰落断电时隙占比随速度升高：24.4% → 26.5% → 39.8%
（贴 p_max 时隙约 0.1% / 0% / 1.0%，说明基本工作在"省电为主"而非"打满"区间）。

## 5. 环境（env/power_control_env.py）

| 项 | 设计 |
|---|---|
| 动作 | 离散 5 档 `[0, 1/4, 1/2, 3/4, 1] × p_max`（5G TPC 风格） |
| 能量截断 | 剩余能量不足时自动降到可支付的最高档，保持离散语义 |
| 观测（10 维） | 最近 8 步 $g$ 归一化 `(g_dB+90)/60`（含当前）+ `t/T` + `E_rem/E` |
| 奖励（默认） | 相对均匀分配的差分奖励 `(rate − rate_uni)·dt·1e-6` |
| 绝对速率 | 评估时用 `info["rate"]` 累加，不用奖励 |

**差分奖励的动机**：不同轨迹信道可差 10 倍，绝对速率的外生方差会淹没"动作优劣"信号；
同轨迹相减消掉外生项，且 $\sum$ 差分 = $R_{agent}-R_{uni}$，后者对给定轨迹是常数，
不改变最优策略。

**当前状态**：环境本体已实现，策略训练与正式评估代码不在仓库内。
接口一致性可用 SB3 一行验证（可选）：

```python
from stable_baselines3.common.env_checker import check_env
check_env(PowerControlEnv("traces/sionna_speed120_test.npz", cfg))   # 未写入仓库
```

## 6. 快速开始

```bash
cd ~/test/mobile-comm-power-allocation
source ~/venvs/rl-pc/bin/activate        # 必须激活！系统 python 没有任何包

python channel/sionna_traces.py     # 重新生成 6 个 npz + sanity 图（可加 --quick）
python -m baselines.selftest        # 5 项检查 + figures/selftest_baselines.png
```

注意：`selftest` 内部用相对路径 `traces/...`，**必须在项目根目录运行**。
`env` 只被环境类引用，没有独立入口脚本。

## 7. 配置（config.py）

改任何参数后**重新运行轨迹生成**即可，下游（env/基线）零改动：

| 参数 | 默认 | 说明 |
|---|---|---|
| `f_c_hz` | 2.0 GHz | 载频；决定多普勒 $f_D=v f_c/c$ |
| `dt_s` / `T` | 1 ms / 1000 | 时隙与 episode 长度 |
| `d_perp_m` | 20 m | 道路到接收节点垂直距离 |
| `speeds_kmh` | (30, 120, 350) | 低/中/高速场景 |
| `shadow_sigma_db` | 4 dB | 每轨迹恒定阴影 |
| `tdl_profile` | "C" | TDL profile（NLOS）；可换 "D"(LOS/Rician) |
| `tdl_delay_spread_s` | 300 ns | TDL-C 名义 RMS 时延扩展 |
| `num_sinusoids` | 20 | 多普勒 SoS 正弦波数 |
| `bandwidth_hz` | 10 MHz | 5G NR FR1 典型带宽 |
| `noise_power` | 1e-8 W | 标定值（中位 SNR ≈ 0 dB，信道受限区） |
| `p_max` / `avg_power` | 0.2 / 0.05 W | 峰值 / 平均功率，比值 4 |
| `n_power_levels` | 5 | 离散档位数（5G TPC 风格） |
| `g_history_len` | 8 | 观测历史窗口长度 |
| `n_train` / `n_test` | 1000 / 100 | 每速度轨迹数 |
| `seed` | 2026 | 测试集固定种子 → 结果可复现 |

**为什么 Δt=1 ms**：2 GHz 下 30 km/h 的相干时间约 7.6 ms，若 Δt=10 ms
连低速信道都会时隙间独立，速度对比失去意义。

## 8. 图表与 notebook 索引

| 脚本 / notebook | 产物 / 用途 |
|---|---|
| `figures/plot_scene.py` | `scene_geometry.pdf`（矢量，供 LaTeX）+ `.png`：场景俯视与 U 形距离 |
| `figures/ray_tracing.py` | `ray_tracing.png`：球面波 $1/r^2$ 扩散 + 建筑反射的**物理示意图**（非射线追踪计算） |
| `baselines/selftest.py` | `figures/selftest_baselines.png`：三速度速率对比 + 示例轨迹信道/功率曲线 |
| `channel/sionna_traces.py` | `results/sionna_sanity_acf.png`（**保留作回归资产**）、`sionna_sanity_timeseries.png`（一次性） |
| `config.ipynb` | 参数选择依据：$f_c$、$\Delta t$、阴影、功率/SNR 标定、数据规模 |
| `channel/sionna_traces.ipynb` | TDL-C 结构、窄带合成、U 形几何、FSPL、单速度生成流程 |
| `channel/trace_io.ipynb` | npz 格式取舍与契约 |
| `baselines/power_alloc.ipynb` | 三行 API 黑箱指南 + 逐函数走读（含 float32 陷阱） |
| `baselines/selftest.ipynb` | 5 项检查各自的原理与判据 |

## 9. 学习路径

按阶段推进，每阶段有"读什么、动手做什么、自检问题"。假设读者已具备通信基础
（懂得 Shannon 容量、路径损耗、衰落）。

### 阶段 0 · 场景与信道

**读**：`config.py` → `channel/trace_io.py` → `channel/sionna_traces.py`，
配合 `config.ipynb` 与 `channel/sionna_traces.ipynb`。

**概念映射**（通信知识 → 代码）：
FSPL 公式在 `fspl_db()`；阴影 = 每轨迹恒定 dB 偏移；TDL-C = 瑞利小尺度；
多普勒通过 `min_speed/max_speed` 进入 Sionna；**$J_0^2(2\pi f_D\Delta t)$ 是功率相关**
（$|h|^2$ 的 lag-1 相关，不是 $h$ 的 $J_0$）。

**动手**：跑 `python channel/sionna_traces.py --quick`，核对打印的 `lag1 实测 ≈ J0² 理论`；
画一条 350 km/h 轨迹，指出 U 形路径损耗弧和快衰落。

**自检**：为什么 Δt=1 ms 而不是 10 ms？相干时间怎么算？

### 阶段 1 · 基线与凸优化

**读**：`baselines/power_alloc.py`（模块文档含完整推导）+ `baselines/selftest.py`，
配合 `baselines/power_alloc.ipynb` / `selftest.ipynb`。

**概念**：凸问题 → KKT 充要 → "水面齐平" → clip 形式
$p^*=\mathrm{clip}(\mu - N/g,\,0,\,p_{\max})$ → 预算方程定水位 μ → 二分求根。
时间维注水 vs 跨子信道注水的区别：这里非因果性来自 μ 依赖**未来**的 $g$。

**动手**：跑 `python -m baselines.selftest`；
画一条轨迹的 $g_t$（dB）与 $p_t^*$ 双轴图，亲眼看到"好信道多分、深衰落断电"；
把 `avg_power` 提到 0.5（进入功率受限区）观察注水增益消失——理解 SNR 区间决定增益。

**自检**：为什么内点时隙边际速率必须相等？为什么注水不是可在线实现的方案？

### 阶段 2 · MDP 形式化

**读**：`env/power_control_env.py`

**概念**：把功率分配写成序列决策——状态 `[g 历史 8 步, t/T, E_rem/E]`、
动作离散 5 档、奖励单时隙速率；硬能量截断的工程含义；
**差分奖励**（相对均匀中心化）：不同轨迹信道差 10 倍，会淹没有效信号，
同轨迹内相减可消掉外生方差，且不改变最优策略。

**动手**：写一个"当前 $g$ 高就发满功率"的朴素策略跑 env，
观察它前期能量烧光、后期断电——理解为什么预算感知是策略的一部分。

**自检**：为什么观测要带 E_rem 和 t？为什么观测含当前完美 $g_t$ 而历史窗口只提供趋势素材？

## 10. 环境与常见坑

| 坑 | 说明 |
|---|---|
| 解释器 | 本项目一律用 `~/venvs/rl-pc/bin/python`（先 `source ~/venvs/rl-pc/bin/activate`）。系统 python 无任何包 |
| CUDA | GPU 代码在自己终端跑；Trae 沙箱内 CUDA 会报 Error 304。轨迹生成/基线/selftest 是 CPU 的，无此限制 |
| 图注乱码 | 信道 sanity 图与几何图内文字用**英文**（默认字体无 CJK）；`selftest` 显式指定 WenQuanYi Zen Hei，可渲染中文 |
| float32 陷阱 | traces 存 float32；基线入口已强转 float64（否则求和误差污染预算判据）。NumPy 2.x 弱标量提升会放大此类问题 |
| 结果目录 | `results/` 未入库；sanity 图需本地跑信道生成才会出现 |
| 未使用依赖 | `stable-baselines3` / `tensorboard` 为后续学习策略阶段预留，当前代码未 import |
| 重建环境 | Python 3.12 + `uv pip install --python ~/venvs/rl-pc/bin/python -r requirements.txt`。torch 官方源易卡死：`uv pip install "torch==2.11.0+cu128" --find-links https://mirrors.aliyun.com/pytorch-wheels/cu128/ --index-url https://pypi.tuna.tsinghua.edu.cn/simple` |
