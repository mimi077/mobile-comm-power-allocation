# power-control-hsr

高速移动车辆 ↔ 静止接收节点的**时隙功率分配**对比实验：平均分配 vs 注水（非因果上界）vs 学习策略（因果），在低/中/高三个速度场景下比较，并研究速度（信道时变性）对三者差距的影响。

## 1. 问题设定

车沿直路经过一个静止接收节点（最近点在 episode 中点），每时隙决策一次发射功率：

- 目标：最大化 episode 总传输量 $\sum_t B\log_2(1+p_t g_t / N_0)\Delta t$
- 约束：硬能量预算 $\sum_t p_t\Delta t \le E$，$0\le p_t\le p_{\max}$
- 信道 $g_t$：外生序列（功率不影响信道），**离线预生成**，训练时查表

| 对比组 | 方法 | 信息条件 |
|---|---|---|
| 平均分配 | $p_t=\bar P$ 恒定 | 无信道信息（下界） |
| 注水 | $p_t^*=\mathrm{clip}(\mu - N/g_t,\,0,\,p_{\max})$ | 非因果，全程 CSI（理论**上界**） |
| 学习策略 | 离散 5 档 TPC 式决策 | 因果（当前/历史 CSI + 剩余能量） |

速度场景：**30 / 120 / 350 km/h**。信道由 Sionna TDL-C（TR 38.901）离线生成，
三组方法读同一批轨迹，严格可比。相邻时隙功率相关 ≈ 0.94 / 0.32 / 0.15（与 J0² 理论吻合）。

## 2. 当前状态与目录结构

```text
power-control-hsr/
├── README.md                    ← 本文件
├── requirements.txt             依赖冻结（rl-pc 环境）
├── config.py                    场景参数唯一来源（所有模块共用）
├── channel/
│   ├── sionna_traces.py    [✓]  Sionna 信道轨迹生成器（TDL-C + FSPL + 4dB 阴影）
│   └── trace_io.py         [✓]  npz 读写（数据契约，下游唯一入口）
├── traces/                 [✓]  6 个 npz：3 速度 × {train, test}
├── baselines/              [✓]  平均分配 + 注水（已打包，见 §3）
│   ├── power_alloc.py           算法本体（模块文档含注水推导）
│   └── selftest.py              正确性自检（预算/KKT/cvxpy 对拍）
├── env/power_control_env.py [✓] Gymnasium 环境（离散 5 档 + 8 步历史 + 差分奖励）
├── run_demo.py             [旧] 连续动作版 demo，待适配离散环境
├── exp_*.py                [实验记录] 排查过程中的临时脚本（学习素材，后续清理）
└── results/                     图表输出
```

**已知待办**：traces 当前为 T=200（练习时重新生成），config.py 仍写 T=1000，
需统一（重跑 `python channel/sionna_traces.py` 约需 10 秒）。

## 3. 基线（baselines 包，已验证）

### 3.1 使用

```python
from baselines import uniform_power, waterfilling, total_rate

p_uni = uniform_power(g, avg_power, p_max)          # 恒定功率，下界
p_wf  = waterfilling(g, avg_power, p_max, noise)    # 闭式二分注水，非因果上界
bits  = total_rate(p_wf, g, bandwidth, noise, dt)   # 总传输量 [bit]
```

主入口 `waterfilling` = 闭式二分解（数值稳定、对该凸问题精确）；
`waterfilling_cvxpy` 保留用于独立对拍。**原理与 KKT 推导见
[baselines/power_alloc.py](baselines/power_alloc.py) 模块文档**。

### 3.2 自检（一条命令回归验证）

```bash
python -m baselines.selftest            # 每速度抽查 20 条测试轨迹
python -m baselines.selftest --traces 50
```

检查项：① 能量预算不超额（含相对容差 1e-10）；② 功率上下界；
③ KKT——内点时隙边际速率相等（"水面齐平"）、贴 p_max 时隙边际更高；
④ 与 cvxpy 独立求解对拍（速率相对差 < 1e-4）；⑤ 逐轨迹注水 ≥ 均匀。

### 3.3 参考数字（全量 100 条测试轨迹，T=200）

| 速度 | 均匀 (Mb) | 注水 (Mb) | 注水增益 |
|---|---|---|---|
| 30 km/h | 2.80 | 3.06 | +9.4% |
| 120 km/h | 2.81 | 3.09 | +9.9% |
| 350 km/h | 2.67 | 2.96 | +10.8% |

注水增益主要来自路径损耗+阴影的全程大动态范围，对速度不敏感。

## 4. 学习策略阶段结论（实验记录，正式 evaluate 待做）

1. **从零 PPO 训不起来**（连续/离散、不同并行度/奖励塑形、250k 步均低于均匀）——
   长时域能量耦合的探索与跨时隙信用分配是瓶颈，实验证据链见 `exp_diag.py` /
   `exp_speeds.py` / `exp_bc.py`
2. **行为克隆（BC）成功**：模仿"保留价格"因果老师（`E_rem/剩余时隙` 自校正基准 +
   机会主义倍乘），三速度稳定超过均匀（`exp_bc2.py`）
3. **瞬时因果策略**（`exp_instant.py`，α=0.5）：完美 CSI 下 +4.4% / +6.0% / +6.6%，
   核心不等式 **均匀 < 学习策略 ≤ 注水** 三速度成立
4. **加入 1ms 过时 CSI 后速度趋势出现**（`exp_delay.py`）：30 km/h 仍 +3.2%，
   120/350 掉到 −6.5% / −8.8%——高速旧 CSI 失相关，对错方向发射
   （测量→上报→生效延迟是高铁场景真实机制）

## 5. 快速开始

```bash
cd ~/power-control-hsr
source ~/venvs/rl-pc/bin/activate        # 必须激活！系统 python 没有任何包

python channel/sionna_traces.py          # 生成轨迹（~10 秒）
python -m baselines.selftest             # 验证基线
python env/power_control_env.py          # env 自检（check_env） #未实现
```

## 6. 配置（config.py）

改任何参数后**重新运行轨迹生成**即可，下游（env/算法）零改动：

| 参数 | 默认 | 说明 |
|---|---|---|
| `f_c_hz` | 2.0 GHz | 载频；决定多普勒 $f_D=v f_c/c$ |
| `dt_s` / `T` | 1 ms / 1000 | 时隙与 episode 长度（当前 traces 为 T=200） |
| `d_perp_m` | 20 m | 道路到接收节点垂直距离 |
| `speeds_kmh` | (30, 120, 350) | 低/中/高速场景 |
| `shadow_sigma_db` | 4 dB | 每轨迹恒定阴影 |
| `tdl_profile` | "C" | TDL profile（NLOS）；可换 "D"(LOS/Rician) |
| `p_max` / `avg_power` | 0.2 / 0.05 W | 峰值 / 平均功率（中位 SNR ≈ −2 dB，信道受限区） |
| `n_power_levels` | 5 | 离散档位数（5G TPC 风格） |
| `n_train` / `n_test` | 1000 / 100 | 每速度轨迹数 |
| `seed` | 2026 | 测试集固定种子 → 结果可复现 |

**为什么 Δt=1 ms**：2 GHz 下 30 km/h 的相干时间约 7.6 ms，若 Δt=10 ms
连低速信道都会时隙间独立，速度对比失去意义。

## 7. 学习路径

按阶段推进，每阶段有"读什么、动手做什么、自检问题"。假设读者已具备通信基础
（懂得 Shannon 容量、路径损耗、衰落），RL 从零学。

### 阶段 0 · 场景与信道（半天）

**读**：`config.py` → `channel/trace_io.py` → `channel/sionna_traces.py`

**概念映射**（通信知识 → 代码）：
FSPL 公式在 `fspl_db()`；阴影 = 每轨迹恒定 dB 偏移；TDL-C = 瑞利小尺度；
多普勒通过 `min_speed/max_speed` 进入 Sionna；**J0²(2πf_D·Δt) 是功率相关**
（|h|² 的 lag-1 相关，不是 h 的 J0）——这是本阶段最容易错的一点。

**动手**：重跑轨迹生成，核对打印的 `lag1 实测 ≈ J0² 理论`；
画一条 350 km/h 轨迹，指出 U 形路径损耗弧和快衰落。

**自检**：为什么 Δt=1 ms 而不是 10 ms？相干时间怎么算？

### 阶段 1 · 基线与凸优化（1 天）

**读**：`baselines/power_alloc.py`（模块文档含完整推导）+ `baselines/selftest.py`

**概念**：凸问题 → KKT 充要 → "水面齐平" → clip 形式
$p^*=\mathrm{clip}(\mu - N/g,\,0,\,p_{\max})$ → 预算方程定水位 μ → 二分求根。
时间维注水 vs 跨子信道注水的区别：这里非因果性来自 μ 依赖**未来**的 g。

**动手**：跑 `python -m baselines.selftest`；
画一条轨迹的 $g_t$（dB）与 $p_t^*$ 双轴图，亲眼看到"好信道多分、深衰落断电"；
把 `avg_power` 提到 0.5（进入功率受限区）观察注水增益消失——理解 SNR 区间决定增益。

**自检**：为什么内点时隙边际速率必须相等？为什么注水不是可在线实现的方案？

### 阶段 2 · MDP 形式化（1 天）

**读**：`env/power_control_env.py`

**概念**：把功率分配写成序列决策——状态 `[g 历史 8 步, t/T, E_rem/E]`、
动作离散 5 档（5G TPC 风格）、奖励单时隙速率；硬能量截断的工程含义；
**差分奖励**（相对均匀中心化）：不同轨迹信道差 10 倍，会淹没有效信号，
同轨迹内相减可消掉外生方差，且不改变最优策略。

**动手**：写一个"当前 g 高就发满功率"的朴素策略跑 env，
观察它前期能量烧光、后期断电——理解为什么预算感知是策略的一部分。

**自检**：为什么观测要带 E_rem 和 t？为什么历史窗口模拟 CSI 滤波？

### 阶段 3 · 学习策略：BC 与 PPO（2-3 天，本项目核心教训）

**读**：`exp_diag.py`（PPO 学习曲线）、`exp_bc.py`（BC 诊断）、
`exp_bc2.py`（成功的 BC）、`exp_instant.py`（瞬时因果策略）

**概念**：BC = 监督学习模仿示范（离散动作 → 分类问题，干净稳定）；
PPO = 策略梯度 + clip + GAE（SB3 一行调用，但调起来不简单）。
**本项目实验链证明**：长时域能量耦合下从零 PPO 学不会（不是超参问题，
是探索+信用分配的结构性困难）；BC + 预算感知老师一次成功；
PPO 热启动后反而把好策略带坏（灾难性遗忘，`exp_warmstart.py`）。

**动手**：复现 `exp_instant.py` 三速度表；扫 α ∈ {0.3, 0.5, 0.8} 找最优。

**自检**：为什么 BC 在这里比从零 PPO 靠谱？什么条件下应该反过来选 PPO？

### 阶段 4 · 实验设计与结论（1-2 天）

**读**：`exp_delay.py`（过时 CSI）

**核心不等式**：均匀 < 因果学习策略 ≤ 注水（非因果上界）。
实验价值 = 量化"因果策略能逼近上界多少（gap），以及 gap 随速度/CSI 时延怎么变"。

**动手**：跑 `exp_delay.py`，理解延迟=0 与延迟=1 两列在三速度下的反转；
画"增益 vs 速度 × CSI 延迟"曲线——这就是本项目的核心图。

**自检**：为什么完美 CSI 下速度越高因果增益反而略大，而过时 CSI 下反转？
（提示：opportunistic 增益只依赖当前 g；预测增益依赖相关性，相关性随速度衰减。）

## 8. 环境与常见坑

| 坑 | 说明 |
|---|---|
| 解释器 | 本项目一律用 `~/venvs/rl-pc/bin/python`（先 `source ~/venvs/rl-pc/bin/activate`）。`/bin/python` 是系统 3.14，无任何包 |
| CUDA | GPU 代码（PPO 训练）在**自己的 WSL 终端**跑；Trae 沙箱内 CUDA 会报 Error 304。轨迹生成/基线/selftest 是 CPU 的，无此限制 |
| torch 安装 | 官方源可能卡死，用阿里云镜像（见 requirements.txt 同目录的安装记录） |
| 图注乱码 | matplotlib 默认字体无中文字形，**图内文字用英文**，终端打印可用中文 |
| float32 陷阱 | traces 存 float32；基线入口已强转 float64（否则求和误差污染预算判据）。NumPy 2.x 弱标量提升会放大此类问题 |
| 重建环境 | Python 3.12 + `uv pip install -r requirements.txt`（torch 用镜像） |
