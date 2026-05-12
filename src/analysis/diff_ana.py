import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# ================== 配置区域 ==================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "results" / "matched_difficulty.csv"
RAW_FILE = PROJECT_ROOT / "data" / "results" / "raw_features_diff.csv"
FIG_DIR = PROJECT_ROOT / "figures" / "analysis"

# 确保输出目录存在
FIG_DIR.mkdir(parents=True, exist_ok=True)
# ============================================

def compute_smd(mean1, std1, mean2, std2):
    """计算标准化均值差异 (SMD)"""
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

def check_balance_and_plot(df_matched, df_raw=None):
    """1. 质量检验：不仅看 Gap，还要算 SMD，并画出诊断图"""
    print("\n=== 1. 匹配质量检验 (Balance Check & Diagnostics) ===")
    
    # 基本 Gap 检查
    diff_gap = (df_matched['diff_t'] - df_matched['diff_c']).mean()
    ps_gap = (df_matched['ps_t'] - df_matched['ps_c']).mean()
    print(f"样本对数: {len(df_matched)}")
    print(f"题目难度平均差异 (Diff Gap): {diff_gap:.6f}")
    print(f"PS分数平均差异 (PS Gap)  : {ps_gap:.6f}")
    
    # 绘制 PS Overlap 密度图
    plt.figure(figsize=(8, 5))
    sns.kdeplot(df_matched['ps_t'], label='Treated (Matched)', fill=True, color='#e74c3c')
    sns.kdeplot(df_matched['ps_c'], label='Control (Matched)', fill=True, color='#3498db')
    plt.title("Propensity Score Overlap After Matching")
    plt.xlabel("Propensity Score")
    plt.legend()
    plt.tight_layout()
    ps_overlap_path = FIG_DIR / "ps_overlap.png"
    plt.savefig(ps_overlap_path, dpi=300)
    plt.close()
    print(f"-> 匹配诊断图已保存: {ps_overlap_path}")

def plot_love_plot(df_matched, df_raw):
    """补齐 8.1 节要求：对所有协变量计算 SMD 并绘制 Love Plot"""
    if df_raw is None:
        return
        
    print("\n=== 1.5 协变量平衡性深度诊断 (Love Plot) ===")
    covariates = ['acc_rate', 'n_count', 'ms_response', 'difficulty']
    
    # 1. 匹配前的全量数据 (Unmatched)
    treated_raw = df_raw[df_raw['hint_used'] == 1]
    control_raw = df_raw[df_raw['hint_used'] == 0]
    
    # 2. 匹配后的保留数据 (Matched)
    # 利用 df_matched 中的 t_idx 和 c_idx 去 raw_df 中精确取回对应的人
    treated_matched = df_raw.iloc[df_matched['t_idx']]
    control_matched = df_raw.iloc[df_matched['c_idx']]
    
    smd_records = []
    for cov in covariates:
        # 计算匹配前的 SMD
        mean_t_raw, std_t_raw = treated_raw[cov].mean(), treated_raw[cov].std()
        mean_c_raw, std_c_raw = control_raw[cov].mean(), control_raw[cov].std()
        smd_unmatched = compute_smd(mean_t_raw, std_t_raw, mean_c_raw, std_c_raw)
        
        # 计算匹配后的 SMD
        mean_t_mat, std_t_mat = treated_matched[cov].mean(), treated_matched[cov].std()
        mean_c_mat, std_c_mat = control_matched[cov].mean(), control_matched[cov].std()
        smd_matched = compute_smd(mean_t_mat, std_t_mat, mean_c_mat, std_c_mat)
        
        smd_records.append({
            'Covariate': cov,
            'Unmatched': abs(smd_unmatched),
            'Matched': abs(smd_matched)
        })
        
    smd_df = pd.DataFrame(smd_records).set_index('Covariate')
    
    # 打印给评委看的数据
    print("匹配后各协变量的绝对标准化均值差异 (|SMD|):")
    for cov, row in smd_df.iterrows():
        status = "✅ 达标" if row['Matched'] < 0.1 else "❌ 超标"
        print(f"   - {cov:<15}: {row['Matched']:.4f} ({status})")
    
    # 开始画 Love Plot
    plt.figure(figsize=(8, 5))
    plt.axvline(x=0.1, color='red', linestyle='--', label='Threshold (0.1)') # 经验阈值线
    
    # 画点
    plt.plot(smd_df['Unmatched'], smd_df.index, 'bo', label='Unmatched (Before)')
    plt.plot(smd_df['Matched'], smd_df.index, 'gs', label='Matched (After)')
    
    # 用浅色线把同一个特征前后的点连起来，视觉冲击力更强
    for i in range(len(smd_df)):
        plt.plot([smd_df['Unmatched'].iloc[i], smd_df['Matched'].iloc[i]], [i, i], 'k-', alpha=0.3)
        
    plt.title("Covariate Balance (Love Plot)")
    plt.xlabel("Absolute Standardized Mean Difference (|SMD|)")
    plt.legend()
    plt.tight_layout()
    
    save_path = FIG_DIR / "love_plot.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"-> 绝杀图表 Love Plot 已保存: {save_path}")
def analyze_simpsons_paradox(df_matched, df_raw):
    """2. 辛普森悖论对比：朴素效应 vs 真实效应"""
    print("\n=== 2. 辛普森悖论分析 (Simpson's Paradox) ===")
    
    if df_raw is None:
        print("未找到原始数据文件，跳过朴素效应计算。")
        return

    # 计算未匹配时的朴素效应
    naive_t = df_raw[df_raw['hint_used'] == 1]['outcome'].mean()
    naive_c = df_raw[df_raw['hint_used'] == 0]['outcome'].mean()
    naive_lift = naive_t - naive_c
    
    # 计算匹配后的真实效应
    matched_t = df_matched['outcome_t'].mean()
    matched_c = df_matched['outcome_c'].mean()
    matched_lift = matched_t - matched_c
    
    print(f"【未控制难度】朴素提升 (Naive Lift): {naive_lift*100:.2f}% (Treated: {naive_t:.4f}, Control: {naive_c:.4f})")
    print(f"【严格控制难度】真实提升 (True ATT) : {matched_lift*100:.2f}% (Treated: {matched_t:.4f}, Control: {matched_c:.4f})")
    
    if (naive_lift > 0 and matched_lift < 0) or abs(naive_lift - matched_lift) > 0.05:
        print("结论：观测到显著的辛普森悖论！提示的整体正向效应是由于难度和倾向性等混杂因素造成的假象。")

def check_selection_bias(df_matched, df_raw):
    """3. 稳健性检验：保留样本 vs 丢弃样本是否存在系统性偏差？"""
    print("\n=== 3. 稳健性检验：选择性偏差 (Selection Bias Check) ===")
    
    if df_raw is None:
        return

    # 提取全体看提示的人
    treated_all = df_raw[df_raw['hint_used'] == 1].reset_index()
    # 假设 matching 时存的是原始行索引，或者可以通过 ps 分数对齐 (为了简化，这里直接通过 ps 匹配或近似认为匹配成功的部分属于 retained)
    # 因为直接对齐索引可能复杂，我们用一个简化的逻辑：匹配成功的样本 PS 分布 vs 全体 Treated 的 PS 分布
    retained_ps = df_matched['ps_t'].values
    
    print(f"Treated 总人数: {len(treated_all)}，保留人数: {len(retained_ps)}")
    print(f"丢弃率: {(1 - len(retained_ps)/len(treated_all))*100:.2f}%")
    
    mean_all_ps, std_all_ps = treated_all['ps'].mean(), treated_all['ps'].std()
    mean_ret_ps, std_ret_ps = df_matched['ps_t'].mean(), df_matched['ps_t'].std()
    smd_retention = compute_smd(mean_ret_ps, std_ret_ps, mean_all_ps, std_all_ps)
    
    print(f"保留样本与总体样本的 PS 得分 SMD: {smd_retention:.4f}")
    if abs(smd_retention) > 0.1:
        print("【严重警告】|SMD| > 0.1！保留下来的样本不能代表总体，模型外部效度（External Validity）受限。")
        print("-> 建议在答辩时说明：当前结论仅适用于特征落在Common Support区域的这部分学生。")
    else:
        print("检验通过：丢弃操作没有引发严重的系统性偏差，结论具有较强泛化能力。")

def analyze_heterogeneity(df):
    """4. 异质性分析：不同难度下的效果差异"""
    print("\n=== 4. 难度异质性分析 (Heterogeneity by Difficulty) ===")
    
    def categorize_diff(x):
        if x >= 0.7: return 'Easy'
        elif x <= 0.4: return 'Hard'
        else: return 'Medium'
    
    df['Difficulty_Group'] = df['diff_t'].apply(categorize_diff)
    
    groups = []
    for name, group in df.groupby('Difficulty_Group'):
        mean_t = group['outcome_t'].mean()
        mean_c = group['outcome_c'].mean()
        lift = mean_t - mean_c
        _, p_val = stats.ttest_rel(group['outcome_t'], group['outcome_c'])
        
        groups.append({
            'Group': name, 'Count': len(group),
            'With Hint': mean_t, 'No Hint': mean_c,
            'Lift': lift, 'P-value': p_val
        })
    
    res_df = pd.DataFrame(groups).set_index('Group')
    order = ['Hard', 'Medium', 'Easy']
    res_df = res_df.reindex(order)
    print(res_df)
    return res_df

def plot_results(df, res_summary):
    """5. 绘制异质性柱状图 (修复了 Seaborn 警告)"""
    print(f"\n正在绘图至 {FIG_DIR}...")
    
    plot_df = df[['diff_t', 'outcome_t', 'outcome_c']].copy()
    plot_df['Difficulty_Group'] = plot_df['diff_t'].apply(
        lambda x: 'Easy' if x >= 0.7 else ('Hard' if x <= 0.4 else 'Medium')
    )
    
    df_melted = plot_df.melt(
        id_vars=['Difficulty_Group'], value_vars=['outcome_t', 'outcome_c'],
        var_name='Condition', value_name='Accuracy'
    )
    df_melted['Condition'] = df_melted['Condition'].map({'outcome_t': 'With Hint', 'outcome_c': 'No Hint'})
    
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(10, 6))
    
    order = ['Hard', 'Medium', 'Easy']
    # 修复：去掉了弃用的 errwidth，改用 err_kws
    ax = sns.barplot(
        data=df_melted, x='Difficulty_Group', y='Accuracy', hue='Condition',
        order=order, palette=['#e74c3c', '#2ecc71'], capsize=.1,
        err_kws={'linewidth': 1.5}
    )
    
    for i, group_name in enumerate(order):
        lift = res_summary.loc[group_name, 'Lift']
        p_val = res_summary.loc[group_name, 'P-value']
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        y_pos = max(res_summary.loc[group_name, 'With Hint'], res_summary.loc[group_name, 'No Hint']) + 0.05
        if y_pos > 1.0: y_pos = 0.95
        plt.text(i, y_pos, f"{lift*100:+.1f}%\n({sig})", ha='center', va='bottom', color='black', fontweight='bold')

    plt.title('Impact of Hints by Question Difficulty (1:1 Matched)', pad=20)
    plt.ylim(0, 1.15)
    plt.ylabel('Average Accuracy')
    plt.xlabel('Difficulty Level')
    plt.legend(title='Condition', loc='upper left')
    
    save_path = FIG_DIR / "hint_impact_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图片已保存: {save_path}")

def main():
    if not INPUT_FILE.exists():
        print(f"文件不存在: {INPUT_FILE}")
        return
        
    df_matched = pd.read_csv(INPUT_FILE)
    df_raw = pd.read_csv(RAW_FILE) if RAW_FILE.exists() else None

    if df_raw is None:
        print("【提示】缺少原始特征文件 raw_features_diff.csv，无法执行辛普森悖论和选择性偏差检验。")
        print("请参考指导，在 matching_diff.py 中保存提取好的特征矩阵。")

    check_balance_and_plot(df_matched, df_raw)
    plot_love_plot(df_matched, df_raw)
    analyze_simpsons_paradox(df_matched, df_raw)
    check_selection_bias(df_matched, df_raw)
    summary = analyze_heterogeneity(df_matched)
    plot_results(df_matched, summary)
    
    print("\n分析与检验全流程执行完毕！可以直接截图相关数据准备答辩或撰写报告了。")

if __name__ == "__main__":
    main()