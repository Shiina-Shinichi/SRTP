import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ================== 配置区域 ==================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "results" / "matched_difficulty.csv"
FIG_DIR = PROJECT_ROOT / "figures" / "analysis"
FIG_DIR.mkdir(parents=True, exist_ok=True)
# ============================================

def categorize_diff(x):
    # 统一使用 0.7 和 0.4 作为阈值（可按需调整）
    if x >= 0.7: return 'Easy'
    elif x <= 0.4: return 'Hard'
    else: return 'Medium'

def main():
    if not INPUT_FILE.exists():
        print(f"找不到文件: {INPUT_FILE}")
        return
        
    df = pd.read_csv(INPUT_FILE)
    print(f"成功加载 1:1 匹配数据: {len(df)} 对")
    
    # 1. 构建二维网格标签
    df['Diff_Group'] = df['diff_t'].apply(categorize_diff)
    
    try:
        df['PS_Group'] = pd.qcut(df['ps_t'], 3, labels=['Low PS', 'Mid PS', 'High PS'])
    except ValueError:
        print("PS分数分布过于集中，退化为两组拆分...")
        df['PS_Group'] = pd.qcut(df['ps_t'], 2, labels=['Low PS', 'High PS'])

    # 2. 计算各个交叉网格中的：正确率 & 样本量
    # 计算正确率均值
    pivot_t = pd.pivot_table(df, values='outcome_t', index='Diff_Group', columns='PS_Group', aggfunc='mean')
    pivot_c = pd.pivot_table(df, values='outcome_c', index='Diff_Group', columns='PS_Group', aggfunc='mean')
    # 计算样本量 (由于是 1:1 匹配，算 outcome_t 的 count 就等于配对的数量 N)
    pivot_count = pd.pivot_table(df, values='outcome_t', index='Diff_Group', columns='PS_Group', aggfunc='count')
    
    # 3. 计算因果效应矩阵 (Lift)
    pivot_lift = pivot_t - pivot_c
    
    # 重新排序（符合直觉：难度从上往下递减）
    order = ['Hard', 'Medium', 'Easy']
    pivot_t = pivot_t.reindex(order)
    pivot_c = pivot_c.reindex(order)
    pivot_lift = pivot_lift.reindex(order)
    pivot_count = pivot_count.reindex(order)

    # 4. 构建自定义格式的注释矩阵 (将百分比和N拼接在一起)
    annot_c = pivot_c.copy().astype(str)
    annot_t = pivot_t.copy().astype(str)
    annot_lift = pivot_lift.copy().astype(str)

    for i in pivot_lift.index:
        for j in pivot_lift.columns:
            # 获取该格子的样本对数，并处理可能出现的 NaN (如果某个格子没人)
            cnt = int(pivot_count.loc[i, j]) if pd.notna(pivot_count.loc[i, j]) else 0
            
            if cnt > 0:
                annot_c.loc[i, j] = f"{pivot_c.loc[i, j]:.1%}\n(N={cnt})"
                annot_t.loc[i, j] = f"{pivot_t.loc[i, j]:.1%}\n(N={cnt})"
                annot_lift.loc[i, j] = f"{pivot_lift.loc[i, j]:+.2%}\n(N={cnt})"
            else:
                annot_c.loc[i, j] = "N/A\n(N=0)"
                annot_t.loc[i, j] = "N/A\n(N=0)"
                annot_lift.loc[i, j] = "N/A\n(N=0)"

    # ================== 控制台打印 ==================
    print("\n=== 交叉异质性：因果提升矩阵 (Lift) ===")
    print(pivot_lift.map(lambda x: f"{x:+.2%}"))
    print("\n=== 各网格样本量 (N) ===")
    print(pivot_count.map(lambda x: f"{int(x)} 对" if pd.notna(x) else "0 对"))

    # ================== 绘制高颜值热力图 ==================
    plt.figure(figsize=(18, 6))  # 稍微增加一点高度，容纳换行的文字
    sns.set_theme(style="white", font_scale=1.1)

    # 注意这里的 fmt=""，因为我们已经自己把字符串格式化好了
    # 子图1：没看提示的正确率基线
    plt.subplot(1, 3, 1)
    sns.heatmap(pivot_c, annot=annot_c, fmt="", cmap="Blues", cbar=False, linewidths=.5)
    plt.title("Control Accuracy (Baseline)")
    plt.ylabel("Question Difficulty")
    plt.xlabel("Student Persona (PS Score)")

    # 子图2：看了提示的正确率
    plt.subplot(1, 3, 2)
    sns.heatmap(pivot_t, annot=annot_t, fmt="", cmap="Blues", cbar=False, linewidths=.5)
    plt.title("Treated Accuracy (With Hint)")
    plt.ylabel("")
    plt.xlabel("Student Persona (PS Score)")

    # 子图3：核心结论 —— 异质性因果效应
    plt.subplot(1, 3, 3)
    sns.heatmap(pivot_lift, annot=annot_lift, fmt="", cmap="RdYlBu", center=0, cbar_kws={'label': 'Accuracy Lift'}, linewidths=.5)
    plt.title("Causal Lift Matrix (Treatment Effect)")
    plt.ylabel("")
    plt.xlabel("Student Persona (PS Score)")

    plt.tight_layout()
    save_path = FIG_DIR / "2d_cross_heterogeneity.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ 交叉异质性热力图（含样本量）已生成并保存至: {save_path}")

if __name__ == "__main__":
    main()