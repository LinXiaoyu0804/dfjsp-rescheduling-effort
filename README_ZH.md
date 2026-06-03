# 响应时间约束下动态柔性作业车间重调度的力度决策

**可实现价值上限与学习适用性的实证研究** —— 论文配套代码与实验仓库。

> 英文说明见 [README_EN.md](README_EN.md)；论文章节↔代码↔表/图的完整映射见 [PAPER_MAP.md](PAPER_MAP.md)。

## 研究问题

在动态柔性作业车间（DFJSP）扰动后的**预测—反应式重调度**中，本文考察一个前置问题：在**增量约束、响应时限、且配有精确局部修复后端**的设定下，在线决定"修复多大一块计划"（重调度力度）相对"最小右移"这一平凡基线，究竟能带来多少**可实现**的改进？

为使其严格可比，我们把在线决策抽象为对**重调度力度**的选择，构造共用同一 CP-SAT 后端的单调力度阶梯：

- **L0 最小右移**：仅移位恢复可行性，不调用求解器，最快、几乎永远可行；
- **L1 小邻域**：释放直接前驱/后继与可选机邻居；
- **L2 影响簇**：进一步释放沿路由的传播段与受影响机器（M3/M4 motif）；
- **L3 整体重开**：释放整个活动窗口，近似整体重优化。

四档共用同一求解器、同一稳定性机制、同一响应预算 B_t，唯一差别是释放集规模，从而把"修复力度"与"所用算子"两个维度解耦。

## 主要结论

在 Brandimarte（Mk6–Mk10）与三组合成实例、三类扰动、五档预算、约 30 倍 ρ_t 范围（约 1.3×10⁴ 事件）上系统验证，得到**双机制**图景：

1. **低扰动**下力度对全局目标的杠杆结构性地小（命题 1）：四档复合目标极差恒 < 0.13%，事后 oracle 相对 L0 仅约 0.03%，诚实可实现收益虽显著但可忽略（约 10⁻³% 量级）；
2. **高扰动**下激进修复在响应预算内不可行（**可行性墙**），可选项收敛到最小右移。

二者之间不存在"既显著有效、又在预算内可行"的升级工作点，故在整个可实现范围内**力度选择缺乏可实现价值**，**最小右移是稳健可行的默认**。价值更可能存在于离线主动鲁棒层。

## 仓库结构

```
src/            核心库
  scheduling/   力度阶梯(L0-L3)、复合目标、活动窗口、incumbent、ρ_t
  solver/       CP-SAT 精确修复后端、求解时延计量
  baselines/    L0(右移)、L3(整体重开)、MWKR/ATC、DANIEL、DDPG、学习式选择器
  eval/         oracle上限与可实现性评估、γ敏感性、ρ边界、外部基线、指标
  events/ env/  动态扰动生成与重调度环境
  motifs/       L2 影响簇 motif 抽取
  data/ graph/ utils/
scripts/        14 个论文实验入口（见下）
configs/        实例、环境、求解器、基线配置
tests/          单元测试
outputs/        论文产物与冻结的事件轨迹（见下）
```

`outputs/` 已随仓库发布以下**论文产物**与**冻结输入**：

- `episodes/`：论文使用的冻结事件轨迹（Brandimarte held-out + 合成 30×10/50×15/100×20），按种子可完全复现；
- `intensity_grid/`、`intensity_grid_decomp/`：§7.2–7.4 力度网格与目标分量分解；
- `sensitivity/`：§7.5 图 F1–F4、γ 敏感性表与统计检验；
- `rho_boundary/`：§7.6 R0–R5 可行性墙与 headroom 曲线；
- `external_baselines/ddpg/`：已训练的 DDPG 基线 checkpoint 与训练数据。

## 环境

推荐 `Python 3.10/3.11`：

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

关键依赖：`ortools`（CP-SAT 后端）、`torch`（学习式基线）、`numpy/pandas/scipy/matplotlib/seaborn`。

## 复现

仓库随附：冻结的事件轨迹 `outputs/episodes/`、论文图（`outputs/sensitivity/F1–F4`、`outputs/rho_boundary/*.png`）、各汇总表 CSV，以及已训练的 DDPG checkpoint。逐事件的力度网格 jsonl 体积大、可由流水线确定性重算，故未随仓库发布（见 `.gitignore`）。

```bash
# §7.2–7.5 力度网格 → oracle 上限/可实现性/γ 敏感性 → 图 F1–F4、表 3–7
python scripts/run_intensity_grid.py            # 重算网格 → outputs/intensity_grid(_decomp)
python scripts/analyze_intensity_sensitivity.py # → outputs/sensitivity（图与统计）

# §7.6 ρ 边界与可行性墙 → 表 8、图 5/6（自包含，使用随附 episodes/incumbents）
python scripts/run_rho_boundary_experiment.py

# §7.7 外部基线同口径对照 → 表 9（DDPG checkpoint 已随仓库发布）
python scripts/evaluate_external_baselines.py \
  --config configs/default.yaml configs/env/formal_dynamic_stronger_v2.yaml \
           configs/solver/cp_repair_default.yaml configs/baselines/ddpg.yaml \
           configs/baselines/learned_rule_selector.yaml \
  --baselines heuristic_rh dispatching_mwkr dispatching_atc full_reoptimization daniel_local ddpg \
  --eval-episodes-dir outputs/episodes/brandimarte_heldout/episodes \
  --output-dir outputs/external_baselines/table9
```

> 只想查看论文图表：直接打开随附的 `outputs/sensitivity/`（F1–F4 + 汇总/统计 CSV）与 `outputs/rho_boundary/`（图 + 汇总 CSV）。
>
> **完全从原始实例重建**（含事件轨迹与监督数据）的完整流水线顺序见 [PAPER_MAP.md](PAPER_MAP.md) 第 1 节（`generate_synthetic_instances → build_incumbents → generate_episodes → build_state_snapshots → run_intensity_grid → analyze_intensity_sensitivity → run_rho_boundary_experiment`；学习基线另加 `generate_teacher_traces → build_operation_dataset → train_ddpg_baseline / train_learned_rule_selector`）。

## 测试

```bash
python -m unittest discover -s tests
```

（需在已安装 `torch`/`ortools` 的环境中运行；详见 [PAPER_MAP.md](PAPER_MAP.md) 第 4 节测试↔论文要点。）

## 引用

```
@article{rescheduling_effort_dfjsp,
  title  = {响应时间约束下动态柔性作业车间重调度的力度决策：可实现价值上限与学习适用性的实证研究},
  journal= {系统工程学报 (Journal of Systems Engineering)},
  year   = {2026}
}
```
