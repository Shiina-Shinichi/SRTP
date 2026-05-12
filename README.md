# SRTP（工程协作版 / 异质性分析重点）

本仓库用于在 KT3 交互日志（enter/respond 等事件）上开展异质性效应分析(Heterogeneous Treatment Effects)：  
比较“看提示/学习行为（Treatment）”对答题正确率（Outcome）的影响，并研究该影响在不同人群/不同题目难度下是否存在显著差异。

> 语言：Python（100%）

---

## 1. 实验目标

这是一个观测数据（非随机对照实验）上的因果推断/因果对比问题：

- **处理变量 Treatment (W)**：答题前的一段窗口期内，是否发生“学习行为”（进入 explanation/lecture），代码里称 `hint_used ∈ {0,1}`。
- **结果变量 Outcome (Y)**：该次答题是否正确，代码里称 `outcome ∈ {0,1}`，通过 `questions.csv` 的标准答案对照得到。
- **协变量 Covariates (X)**：用于控制混杂因素，使 treated/control 尽量可比（能力、经验、速度等的代理变量，如历史正确率、做题次数、耗时、题目难度）。

### 重点：异质性分析维度（当前仓库已覆盖/部分覆盖）
1. **按题目难度分组**（Hard / Medium / Easy）  
2. **按倾向性得分 PS 分组**（Low / Mid / High，近似“不爱看提示 / 一般 / 爱看提示”的人群）

最终希望产出：
- 可复现的匹配结果数据集（CSV）
- 各分组下 treated/control 的正确率、lift（提升百分点）、显著性/不确定性指标
- 可视化图表（用于报告/论文）

---

## 2. 目录结构

```
.
├── .gitignore
└── src
    ├── processing
    │   └── precompute_diff.py
    ├── matching
    │   ├── matching_hint.py
    │   └── matching_diff.py
    ├── analysis
    │   ├── diff_ana.py
    │   ├── hint_ana.py
    │   ├── vis_diff.py
    │   ├── vis_hint.py
    │   ├── weighting_ana.py    # (新增) 稳健性：重叠加权检验
    │   ├── vis_2d_hte.py       # (新增) 2D交叉异质性：生成难度×人群热力图
    │   └── deep_dive_hte.py    # (新增) 行为学深挖：显著性检验与答题耗时探测
    └── debug
        └── diff_debug.py

```

### 2.1 `src/processing/precompute_diff.py`
- 功能：生成/预计算题目难度表（difficulty 链路使用）。
- 预期产物（供 `matching_diff.py` 读取）：
  - `data/processed/question_difficulty.csv`
  - 字段期望：`item_id, avg_correctness`
    - `avg_correctness` 越高表示题越容易

> 说明：如果难度表缺失，`matching_diff.py` 会给 difficulty 默认值 0.5，但难度对齐会变弱。

### 2.2 `src/matching/matching_hint.py`（Hint 链路：PS + 1:k 匹配）
- 功能：
  1) 流式解析 KT3 日志，构造样本 `(X, W, Y)` 
  注意一定要使用流式解压，参考已经可以稳定运行的解压方式，否则会爆磁盘 
  2) 用 LightGBM 训练倾向性得分模型 `PS = P(W=1|X)`  
  3) 在 PS 空间做匹配（FAISS 最近邻，Control → Treated，默认 1:4，带 caliper）  
  4) 保存匹配结果 CSV
- 输出：
  - `data/results/matched_kt3_hint.csv`

### 2.3 `src/matching/matching_diff.py`（Difficulty 链路：PS + 双卡尺 + 1:1）
- 功能：在特征中加入题目难度，并做更“公平”的对齐匹配。
- 算法特点：
  - 用 FAISS 先按 PS 找候选近邻（top `SEARCH_CANDIDATES`）
  - 用两个卡尺过滤：
    1) PS 卡尺：`PS_CALIPER`
    2) 难度卡尺：`DIFF_CALIPER`
  - 强制 **1:1**：Treated 不重复使用（用 `used_treated_indices` 控制）

- 输出：
  - `data/results/matched_difficulty.csv`

### 2.4 `src/analysis/hint_ana.py`（PS 分箱的人群异质性）
- 输入：`data/results/matched_kt3_hint.csv`
- 输出/功能：
  - 总体 treated vs control 正确率差与 lift
  - 简单匹配质量指标（`dist` 均值）
  - 当 `ps_t` 足够离散时，用 `pd.qcut(ps_t, 3)` 分 Low/Mid/High 三组，做分组异质性统计

### 2.5 `src/analysis/diff_ana.py`（难度异质性 + 统计检验 + 绘图）
- 输入：`data/results/matched_difficulty.csv`
- 功能：
  - 简版平衡性检查
  - 总体效应（lift/ATT）+ paired t-test（`stats.ttest_rel`）
  - **按题目难度分组** Hard/Medium/Easy：组内 lift + 组内 paired t-test
  - 输出图到 `figures/analysis/`

### 2.6 `src/analysis/vis_hint.py`（Hint 链路可视化）
- 输入：`data/results/matched_kt3_hint.csv`
- 输出（到 `figures/`）：
  1) 匹配后 PS 分布对比图（balance check 直观展示）`1_ps_distribution.png`
  2) PS 分组异质性正确率对比 `2_group_accuracy.png`
  3) PS 分组 lift（提升）柱状图 `3_net_lift.png`

### 2.7 `src/analysis/vis_diff.py`（Difficulty 链路可视化）
- 输入：脚本里默认是 `data/results/matched_kt3_diff.csv`（请核对；若你们实际文件是 `matched_difficulty.csv` 需要改路径）
- 功能：按难度分组画正确率对比，并打印 lift。

### 2.8 `src/debug/diff_debug.py`（difficulty 异质性排查工具）
- 输入：`data/results/matched_difficulty.csv`
- 功能：
  - 提取 Hard 组子集
  - 检查异常（如 outcome 全相同、PS 完全一致、自匹配等）
  - 打印样本预览，帮助定位问题

### 2.9 `src/analysis/weighting_ana.py`（稳健性检验：重叠加权）
- 输入：`data/results/raw_features_diff.csv`（全量特征数据集，因为加权需要用到被 1:1 匹配丢弃的所有样本）
- 功能：使用 Overlap Weighting (OW) 算法保留 100% 样本，计算加权后的 treated/control 正确率与总体 Lift，并输出加权 SMD 诊断指标，用于验证 1:1 匹配严格丢弃样本是否引发了核心结论的偏移。

### 2.10 `src/analysis/vis_2d_hte.py`（二维交叉异质性热力图）
- 输入：`data/results/matched_difficulty.csv`（1:1 双卡尺匹配后的纯净数据集）
- 功能：将人群画像（PS 分数分为 Low/Mid/High）与题目难度（Hard/Medium/Easy）进行 3×3 交叉，计算每个网格内的独立 Lift，并自动生成带有样本量 (N) 和百分比的高颜值学术热力图 `2d_cross_heterogeneity.png`。

### 2.11 `src/analysis/deep_dive_hte.py`（行为机理深挖与显著性检验）
- 输入：同时依赖 `data/results/matched_difficulty.csv`（获取对齐的 ID）和 `data/results/raw_features_diff.csv`（根据 ID 穿透回溯原始特征中的 `ms_response` 答题耗时对数）。
- 功能：对二维交叉矩阵的每一个网格独立执行配对 t 检验（Paired t-test），在控制台输出 P-value 及显著性星号（***）；同时换算出干预组看提示后的“平均真实答题耗时（秒）”，用行为学数据实锤高 PS 群体在难题上的“系统博弈（Gaming the System）”盲猜行为。
---

## 3. 数据与路径约定（运行前必须对齐）

所有脚本均通过以下方式定位项目根目录：

- `PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent`

默认目录：
- 原始数据：`data/raw/`
- 中间产物：`data/processed/`
- 结果输出：`data/results/`
- 图表输出：`figures/` / `figures/analysis/`

### 必要输入
- `data/raw/kt3.tar.gz`
- `data/raw/questions.csv`
  - 备选：`data/raw/contents/questions.csv`

### difficulty 链路额外输入
- `data/processed/question_difficulty.csv`
  - 期望字段：`item_id, avg_correctness`

---

## 4. 环境准备

建议 Python 3.10+（3.9+ 多数环境也可运行）并创建虚拟环境：

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
```

当前代码用到的依赖：
- numpy, pandas, scipy
- matplotlib, seaborn
- tqdm
- scikit-learn
- lightgbm
- faiss（faiss-cpu / faiss-gpu）

安装示例：

```bash
pip install -U pip
pip install numpy pandas scipy matplotlib seaborn tqdm scikit-learn lightgbm faiss-cpu
```

---

## 5. 快速开始（匹配 → 异质性分析）

### 5.1 Hint 链路（PS + 1:k 匹配）
```bash
python src/matching/matching_hint.py
# 输出：data/results/matched_kt3_hint.csv
```

进行 PS 分组异质性分析与画图：
```bash
python src/analysis/hint_ana.py
python src/analysis/vis_hint.py
# 输出：figures/1_ps_distribution.png
#       figures/2_group_accuracy.png
#       figures/3_net_lift.png
```

### 5.2 Difficulty 链路（PS + 难度卡尺 + 1:1）
确保先准备好：
- `data/processed/question_difficulty.csv`

运行匹配：
```bash
python src/matching/matching_diff.py
# 输出：data/results/matched_difficulty.csv
```

进行难度异质性分析与画图：
```bash
python src/analysis/diff_ana.py
# 输出：figures/analysis/hint_impact_analysis.png（文件名由脚本内部决定）
```

如 Hard 组结果看起来异常，辅助排查：
```bash
python src/debug/diff_debug.py
```
图像大概是一个类似抛物线或者悬链线的样子，这是论文里也提及过的理想的情况。

### 5.3 进阶分析与稳健性检验

在跑完 5.2 后，依次执行以下脚本：

# 1. 稳健性检验 (验证100%样本保留下的结论)
python src/analysis/weighting_ana.py

# 2. 二维交叉异质性 (输出 3x3 热力图)
python src/analysis/vis_2d_hte.py

# 3. 行为机制探究 (输出各组耗时，实锤“刷提示”现象)
python src/analysis/deep_dive_hte.py

---

## 6. 关键函数接口与核心算法说明

### 6.1 Treatment 定义：窗口期判定（Hint Window）
两条 matching 链路里 `hint_used` 的核心逻辑一致：

- 维护 `learning_timestamps`：用户进入 explanation/lecture（`action_type == 'enter'` 且 `item_id` 以 `e` 或 `l` 开头）的时间戳列表
- 当出现答题事件（`action_type == 'respond'` 且 `item_id` 以 `q` 开头）时：
  - 若存在某次 `enter` 满足：`0 <= respond_ts - enter_ts <= HINT_WINDOW_MS`  
    则 `hint_used = 1`，否则为 0

关键参数：
- `HINT_WINDOW_MS`（默认 10 分钟）

### 6.2 倾向性得分 PS 建模（Propensity Score）
- 输入：协变量 X
  - hint 链路常用 3 维：`acc_rate, log1p(n_count), log1p(ms_response)`
  - difficulty 链路 4 维：在上面基础上加入 `difficulty`
- 流程：
  - `StandardScaler` 标准化
  - `LGBMClassifier` 拟合 `W`
  - 得到 `ps_scores = P(W=1 | X)`

### 6.3 匹配算法（Matching）
#### Hint 链路：`matching_hint.py`
- 在 1D PS 空间做最近邻：
  - 默认方向：Control → Treated
  - 默认比例：1:4（`N_NEIGHBORS=4`）
- 使用 `PS_CALIPER` 过滤 PS 差距过大的匹配对
- 输出包含 `treated_idx, control_idx, ps_t, ps_c, dist, rank, outcome_t, outcome_c`

#### Difficulty 链路：`matching_diff.py`
- 先按 PS 找 K 个候选近邻（`SEARCH_CANDIDATES`）
- 双卡尺过滤（同时满足）：
  1) `PS`：`dist_sq <= PS_CALIPER**2`
  2) `difficulty`：`abs(diff_t - diff_c) <= DIFF_CALIPER`
- 强制 1:1：treated 不复用（`used_treated_indices`）

---

## 7. 关键参数（影响异质性结论，修改需记录）

通用：
- `HINT_WINDOW_MS`：窗口期（决定 Treatment 判定）
- `LIMIT_FILES`：是否抽样跑（调试用；会影响最终结论）
- `PS_CALIPER`：PS 卡尺（太严会丢样本，太松会不平衡）

Hint 链路：
- `N_NEIGHBORS`：1:k 匹配比例

Difficulty 链路：
- `DIFF_CALIPER`：难度卡尺（影响“同难度对齐”的强度）
- `SEARCH_CANDIDATES`：候选邻居数（越大越可能找到满足难度卡尺的匹配）

协作规则：
- 任何人改以上参数，必须在 commit/PR 描述中说明“为什么改 + 预期影响”。

---

## 8. 建议与后续计划

### 8.1 加强匹配质量诊断（否则异质性结论站不住）（ 已落实）
当前平衡性检查偏简化，建议补齐标准诊断（最好做到每次实验自动输出一份表+图）：
- 对所有协变量（例如 `acc_rate / n_count / ms_response / difficulty` 等）计算 **SMD（Standardized Mean Difference）**，经验阈值：`|SMD| < 0.1`（越小越好）
- 画 **Love plot**：before/after matching 的 SMD 对比
- 画 **PS overlap**：匹配后 treated/control 的 PS 分布重叠（密度图或直方图）
- 记录样本利用情况：匹配成功对数、丢弃率、是否存在大量重复使用样本（会影响后续推断）

### 8.2 稳健性对比（已落实）
为了证明异质性结论不是某个方法/某套参数的偶然结果，建议至少做 2–3 种方案对比（同一套分组规则下）：
- **PS 分层（quintile/decile）**：作为简单稳定的 baseline（桶内比较 + 加权汇总）
- **加权方法**：ATT-weighting / IPTW / overlap weighting（很多时候比匹配更稳、样本利用率更高）
- **双重稳健**：AIPW（PS 模型 + outcome 模型，增强稳健性）

对比输出建议统一成一张表（每行一个方法）：平衡性指标（max/mean |SMD|）、有效样本量、总体 lift、各分组 lift。

### 8.3救火方案（当匹配失败、分组样本太少、或诊断不过关时）
> 原则：**先让实验跑通并保持可解释性**，再追求最优。救火改动做完必须立刻复查 8.1 的诊断（SMD/overlap）。

#### 8.3.1 参数微调（首选，改动成本最低）
- **放宽卡尺以提高匹配成功率**（常见救火方向）  
  - `PS_CALIPER` 适度变大：匹配对更多，但风险是平衡性变差  
  - difficulty 链路中 `DIFF_CALIPER` 适度变大：保证每个难度组有样本，但会增加“任务不够可比”的风险  
- **增大候选邻居数**（difficulty 链路常用）  
  - 提高 `SEARCH_CANDIDATES`，让 control 更可能找到同时满足 PS+difficulty 卡尺的 treated  
- **调整匹配比例**  （ 已落实）
  - 将 1:1 改为 1:k（k=2/3/4）以提升稳定性（尤其是某些难度组样本太少时）  
  - 若采用 1:k 或有放回，后续推断建议改用 bootstrap CI

#### 8.3.2 模型重构（当 PS 质量差或平衡性怎么都不过时）
- **增强特征/模型表达能力**  
  - 增加关键交互项/非线性项（例如 `做题次数 × 难度`、`acc_rate` 的分段/非线性）  
  - 或对比更简单的 PS 模型（如 Logistic Regression）作为 sanity check（避免模型过拟合导致 PS 失真）
- **共同支撑（common support）修剪**  
  - 剔除 PS 极端、缺少重叠的样本（例如只保留 PS 在某个区间内的数据），避免“拿不可比的人硬比”  
  - 修剪后必须重新做平衡性诊断，并报告修剪比例

#### 8.3.3 策略切换（当 PSM 很难通过诊断/样本利用率太差时）（ 已落实）
- **改用加权（推荐优先尝试 overlap weighting）**  
  - 当匹配一直丢样本或难以平衡时，加权方法通常更稳、实现也不复杂  
- **分层/分箱替代匹配**  
  - 用 PS 分位分层（quintile/decile），桶内比较 + 加权汇总，作为稳定 baseline

---
## 9. 留言

首先跑通，再考虑结论好不好。细节的东西可能还有不少o_o。