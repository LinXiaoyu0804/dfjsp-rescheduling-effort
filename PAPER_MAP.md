# 论文 ↔ 代码映射 (Paper ↔ Code Map)

本文件给出论文《响应时间约束下动态柔性作业车间重调度的力度决策：可实现价值上限与学习适用性的实证研究》各章节与本仓库代码、配置、产物的对应关系，便于复现与审阅。

> 命名约定：力度阶梯 **L0**（最小右移）⊆ **L1**（小邻域）⊆ **L2**（影响簇）⊆ **L3**（整体重开）。
> 扰动杠杆 **ρ_t = w_t / L_t**。复合目标 **J = α·C_max + β·ΣT_j + γ·I**（基线权重 α=β=1, γ=0.2）。

---

## 1. 复现流水线（执行顺序）

| 步 | 脚本 | 作用 | 产物 |
|----|------|------|------|
| 0 | `scripts/generate_synthetic_instances.py` | 生成合成实例 30×10 / 50×15 / 100×20 | `data/raw/fjsp/synthetic_scaling/` |
| 1 | `scripts/build_incumbents.py` | 离线充裕预算求高质量初始计划 incumbent | `outputs/episodes/<scale>/incumbents/` |
| 2 | `scripts/generate_episodes.py` | 注入三类扰动、按种子冻结事件轨迹 | `outputs/episodes/<scale>/episodes/` |
| 3 | `scripts/build_state_snapshots.py` | 导出逐事件可观测状态快照（特征用） | `*/state_snapshots.jsonl` |
| 4 | `scripts/run_intensity_grid.py` | §7.2–7.4 力度网格：同后端强制评测 L0–L3 | `outputs/intensity_grid/`、`outputs/intensity_grid_decomp/` |
| 5 | `scripts/analyze_intensity_sensitivity.py` | §7.5 γ 重加权 + oracle/可实现/捕获率 + 图 F1–F4 | `outputs/sensitivity/` |
| 6 | `scripts/run_rho_boundary_experiment.py` | §7.6 受控工况 R0–R5、可行性墙、headroom 曲线 | `outputs/rho_boundary/` |
| 7a | `scripts/generate_teacher_traces.py` → `scripts/build_operation_dataset.py` | 构建监督训练数据集（供 §7.7/§7.8 学习基线） | teacher operation dataset `.pt` |
| 7b | `scripts/train_ddpg_baseline.py` / `scripts/train_learned_rule_selector.py` | 训练 §7.7 的 DDPG / §7.8 的学习式选择器基线 | `outputs/external_baselines/.../checkpoints/` |
| 8 | `scripts/evaluate_external_baselines.py` | §7.7 外部基线同口径对照（L0/MWKR/ATC/L3/DANIEL/ddpg/selector） | `<output-dir>/<baseline>_event_metrics.jsonl` |

辅助：`scripts/validate_benchmarks.py` 校验基准实例可解析。

---

## 2. 章节 ↔ 核心代码

| 论文章节 | 代码位置 |
|----------|----------|
| §3 重调度力度决策模型；复合目标式(1)、不稳定性式(2) | `src/scheduling/objectives.py` |
| §3 力度阶梯 L0–L3（单调嵌套释放集 R0⊆R1⊆R2⊆R3） | `src/scheduling/intensity_ladder.py` |
| §3 活动窗口 W_t、incumbent、增量约束 | `src/scheduling/{window.py,incumbent.py,incumbent_builder.py,state_builder.py}` |
| §3 L1–L3 共用的 CP-SAT 精确修复后端（暖启动、信赖域、统一预算 B_t） | `src/solver/cp_repair_solver.py` |
| §3 L0=最小右移、L3=整体重优化 | `src/baselines/heuristic_rh.py`、`src/baselines/full_reopt.py` |
| §4 扰动杠杆界（命题 1）、ρ_t 描述子 | `src/scheduling/rho.py` |
| §5 升级增益与事后 oracle（定义 1，式(3)）、诚实可实现策略（定义 2）、捕获率/可实现性落差、部署判据 1、命题 2 | `src/eval/rho_boundary.py` |
| §5/§7.5 目标可分解重加权、bootstrap 置信区间、配对统计 | `src/eval/intensity_sensitivity.py` |
| §6 动态环境、扰动生成（泊松到达、截断对数正态停机）、混合事件 | `src/events/`、`src/env/dfjsp_env.py` |
| §6 受控工况 R0–R5 扰动 profile | `configs/env/rho_boundary_profiles.yaml`、`src/events/generator.py`（`RHO_INTENSITY_PROFILE_DEFAULTS`） |
| §7.7/§7.8 外部基线适配 | `src/baselines/{dispatching.py(MWKR/ATC),full_reopt.py,daniel_local.py,ddpg.py,learned_rule_selector.py}`、`src/eval/external_baselines.py` |

---

## 3. 表 / 图 ↔ 产物

| 论文资产 | 内容 | 生成脚本 | 产物文件 |
|----------|------|----------|----------|
| 表 1 | 主要记号 | （正文） | — |
| 表 2 | 受控工况 R0–R5 的扰动 profile 与实测 ρ_t | `run_rho_boundary_experiment.py` | `outputs/rho_boundary/rho_boundary_summary.csv`；profile 见 `configs/env/rho_boundary_profiles.yaml` |
| 表 3 | 各力度档复合目标均值与四档极差 | `run_intensity_grid.py` | `outputs/intensity_grid/frontier_summary.csv` |
| 表 4 | 各力度档稳定性/扰动指标 | `run_intensity_grid.py` | `outputs/intensity_grid/frontier_summary.csv` |
| 表 5 | L0 vs L3 预算内可行率与在线时延 | `run_intensity_grid.py` | `outputs/intensity_grid/frontier_summary.csv` |
| 表 6 | oracle 上界、可实现策略、捕获率 | `analyze_intensity_sensitivity.py` | `outputs/sensitivity/sensitivity_summary.csv` |
| 表 7 | 目标权重 γ 敏感性 | `analyze_intensity_sensitivity.py` | `outputs/sensitivity/sensitivity_summary.csv`、`stat_tests.csv` |
| 表 8 | 各 ρ_t 区间逐档可行率与可行档头部空间 | `run_rho_boundary_experiment.py` | `outputs/rho_boundary/rho_boundary_summary.csv`、`frontier_summary.csv` |
| 表 9 | 外部基线同口径对照 | `evaluate_external_baselines.py` | `<output-dir>/<baseline>_event_metrics.jsonl`（聚合自 `src/eval/event_summary.py`） |
| 表 10 | 面向部署的决策汇总 | （正文，综合上述） | — |
| 图 1 | 力度–质量–可行前沿 (F1) | `analyze_intensity_sensitivity.py` | `outputs/sensitivity/F1_intensity_quality_frontier.{svg,png}` |
| 图 2 | within-budget 异质性热图 (F2) | `analyze_intensity_sensitivity.py` | `outputs/sensitivity/F2_heterogeneity_heatmap.{svg,png}` |
| 图 3 | 可实现性落差柱状 (F3) | `analyze_intensity_sensitivity.py` | `outputs/sensitivity/F3_value_gap_bars.{svg,png}` |
| 图 4 | 目标权重敏感性 (F4) | `analyze_intensity_sensitivity.py` | `outputs/sensitivity/F4_gamma_sensitivity.{svg,png}` |
| 图 5 | 可行性墙（可行率 vs ρ_t） | `run_rho_boundary_experiment.py` | `outputs/rho_boundary/capture_vs_rho.png`、`figdata_capture_vs_rho.csv` |
| 图 6 | 可行档间头部空间 vs ρ_t | `run_rho_boundary_experiment.py` | `outputs/rho_boundary/headroom_vs_rho.png`、`figdata_headroom_vs_rho.csv` |

> 每张图旁均附 `figdata_*.csv` 原始数据，便于重绘与核对。

---

## 4. 单元测试 ↔ 论文要点

| 测试 | 覆盖 |
|------|------|
| `tests/test_intensity_ladder.py` | §3 力度阶梯单调性 R0⊆R1⊆R2⊆R3、复现闸（L0≡右移、L3≡整体重优化） |
| `tests/test_rho_boundary.py` | §5 oracle 上界、诚实可实现策略、捕获率 |
| `tests/test_intensity_sensitivity.py` | §7.5 γ 重加权与 bootstrap 统计 |
| `tests/test_external_baselines.py`、`tests/test_daniel_local.py` | §7.7 外部基线适配 |
| `tests/test_episode_generator.py`、`tests/test_event_logging.py`、`tests/test_schedule_metrics.py`、`tests/test_runtime_accounting.py` | §6 事件生成、逐事件日志、调度指标、求解时延计量 |
| `tests/test_motif_extractors.py`、`tests/test_grouped_probing.py` | §3 L2 影响簇（M3/M4 motif）释放集构造 |
| `tests/test_teacher_trace_io.py`、`tests/test_event_summary.py` | §7.7/§7.8 监督数据 IO 与分层汇总 |

运行（需在装有 `torch`/`ortools` 的环境，见 README）：

```bash
python -m unittest discover -s tests
```
