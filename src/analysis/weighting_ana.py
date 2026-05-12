import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ================== 配置区域 ==================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "results" / "raw_features_diff.csv"
FIG_DIR = PROJECT_ROOT / "figures" / "analysis"
FIG_DIR.mkdir(parents=True, exist_ok=True)
# ============================================

def compute_weighted_smd(df, covariates):
    """计算 Overlap Weighting 后的 SMD"""
    smd_records = []
    
    # 提取两组
    treated = df[df['hint_used'] == 1]
    control = df[df['hint_used'] == 0]
    
    for cov in covariates:
        # 匹配前的原始 pooled standard deviation (用于分母，保持量纲一致)
        std_pool_raw = np.sqrt((treated[cov].var() + control[cov].var()) / 2)
        
        # 计算加权均值
        mean_t_w = np.average(treated[cov], weights=treated['ow_weight'])
        mean_c_w = np.average(control[cov], weights=control['ow_weight'])
        
        # 计算加权 SMD
        smd_w = abs(mean_t_w - mean_c_w) / std_pool_raw if std_pool_raw > 0 else 0
        smd_records.append({'Covariate': cov, 'Weighted_SMD': smd_w})
        
    return pd.DataFrame(smd_records)

def main():
    if not RAW_FILE.exists():
        print(f"找不到文件: {RAW_FILE}")
        return
        
    df = pd.read_csv(RAW_FILE)
    print(f"成功加载全量数据: {len(df)} 条")
    
    # 1. 计算 Overlap Weights
    # Treated: weight = 1 - PS; Control: weight = PS
    df['ow_weight'] = np.where(df['hint_used'] == 1, 1 - df['ps'], df['ps'])
    
    # 2. 协变量平衡性检验 (Weighted SMD)
    covariates = ['acc_rate', 'n_count', 'ms_response', 'difficulty']
    smd_df = compute_weighted_smd(df, covariates)
    
    print("\n=== 1. Overlap Weighting 平衡性检验 ===")
    print("各协变量的加权绝对标准化均值差异 (|SMD|):")
    for _, row in smd_df.iterrows():
        status = "✅ 极佳" if row['Weighted_SMD'] < 0.1 else "❌ 超标"
        print(f"   - {row['Covariate']:<15}: {row['Weighted_SMD']:.4f} ({status})")
        
    # 3. 总体加权因果效应 (Weighted ATT/ATE)
    mean_t_outcome = np.average(df[df['hint_used'] == 1]['outcome'], weights=df[df['hint_used'] == 1]['ow_weight'])
    mean_c_outcome = np.average(df[df['hint_used'] == 0]['outcome'], weights=df[df['hint_used'] == 0]['ow_weight'])
    overall_lift = mean_t_outcome - mean_c_outcome
    
    print("\n=== 2. 总体加权提示效应 ===")
    print(f"Treated (加权正确率): {mean_t_outcome:.4f}")
    print(f"Control (加权正确率): {mean_c_outcome:.4f}")
    print(f"提升 (Weighted Lift) : {overall_lift*100:.2f}%")

    # 4. 难度异质性分析 (加权版本)
    print("\n=== 3. 难度异质性分析 (Heterogeneity by Difficulty - Weighted) ===")
    def categorize_diff(x):
        if x >= 0.7: return 'Easy'
        elif x <= 0.4: return 'Hard'
        else: return 'Medium'
    
    df['Difficulty_Group'] = df['difficulty'].apply(categorize_diff)
    
    groups = []
    for name, group in df.groupby('Difficulty_Group'):
        t_subset = group[group['hint_used'] == 1]
        c_subset = group[group['hint_used'] == 0]
        
        # 为了防止某些极端分组内没有样本，加个判断
        if len(t_subset) == 0 or len(c_subset) == 0:
            continue
            
        m_t = np.average(t_subset['outcome'], weights=t_subset['ow_weight'])
        m_c = np.average(c_subset['outcome'], weights=c_subset['ow_weight'])
        
        groups.append({
            'Group': name,
            'Count (Total)': len(group),
            'With Hint (W)': m_t,
            'No Hint (W)': m_c,
            'Lift': m_t - m_c
        })
        
    res_df = pd.DataFrame(groups).set_index('Group').reindex(['Hard', 'Medium', 'Easy'])
    print(res_df)
    
    print("\n[注]：你可以将此表的结果与 1:1 匹配的结果放在一起，构建 README 8.2 所要求的稳健性对比表！")

if __name__ == "__main__":
    main()