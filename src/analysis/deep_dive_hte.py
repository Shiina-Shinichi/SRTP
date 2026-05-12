import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# ================== 配置区域 ==================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MATCHED_FILE = PROJECT_ROOT / "data" / "results" / "matched_difficulty.csv"
RAW_FILE = PROJECT_ROOT / "data" / "results" / "raw_features_diff.csv"
# ============================================

def categorize_diff(x):
    if x >= 0.7: return 'Easy'
    elif x <= 0.4: return 'Hard'
    else: return 'Medium'

def main():
    if not MATCHED_FILE.exists() or not RAW_FILE.exists():
        print("找不到数据文件，请确保 MATCHED_FILE 和 RAW_FILE 都已生成。")
        return

    df_matched = pd.read_csv(MATCHED_FILE)
    df_raw = pd.read_csv(RAW_FILE)

    # 1. 获取看提示学生（Treated）的实际答题耗时
    # raw 数据中的 ms_response 是经过 np.log1p 处理的毫秒数，我们需要还原成真实的秒数
    ms_response_log = df_raw.iloc[df_matched['t_idx']]['ms_response'].values
    df_matched['response_time_sec'] = np.expm1(ms_response_log) / 1000.0  # 转换为秒

    # 2. 划分网格
    df_matched['Diff_Group'] = df_matched['diff_t'].apply(categorize_diff)
    # 使用 qcut 划分，避免报错
    df_matched['PS_Group'] = pd.qcut(df_matched['ps_t'], 3, labels=['Low PS', 'Mid PS', 'High PS'])

    # 3. 统计并打印交叉表格
    diff_orders = ['Hard', 'Medium', 'Easy']
    ps_orders = ['Low PS', 'Mid PS', 'High PS']

    print("="*85)
    print(" 🌟 交叉异质性深度探测：显著性检验 (P-value) 与 行为学耗时 (Gaming the System)")
    print("="*85)
    
    for d in diff_orders:
        print(f"\n【题目难度: {d}】")
        print(f"{'人群画像':<12} | {'样本对数':<8} | {'因果提升(Lift)':<12} | {'P-Value':<8} | {'显著性':<5} | {'看提示后平均答题耗时'}")
        print("-" * 85)
        for p in ps_orders:
            subset = df_matched[(df_matched['Diff_Group'] == d) & (df_matched['PS_Group'] == p)]
            if len(subset) == 0:
                continue
                
            # 计算 Lift
            lift = subset['outcome_t'].mean() - subset['outcome_c'].mean()
            
            # 配对 t 检验
            t_stat, p_val = stats.ttest_rel(subset['outcome_t'], subset['outcome_c'])
            
            # 打上显著性星号
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
            
            # 计算看提示人群的平均答题耗时
            avg_time = subset['response_time_sec'].mean()
            
            # 格式化输出
            print(f"{p:<14} | {len(subset):<10} | {lift*100:>8.2f}%    | {p_val:<8.4f} | {sig:<6} | {avg_time:>6.1f} 秒")

if __name__ == "__main__":
    main()