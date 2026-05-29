import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import scipy.stats as stats
import argparse

def calculate_tpr_at_fpr(wm_csv_path, no_wm_csv_path, target_fpr=0.01, method='bimark', target_length=None):
    print(f"读取水印数据: {wm_csv_path}")
    print(f"读取无水印数据: {no_wm_csv_path}")
    
    df_wm = pd.read_csv(wm_csv_path)
    df_no_wm = pd.read_csv(no_wm_csv_path)

    if target_length is not None:
        df_wm = df_wm[df_wm['length'] == target_length]
        df_no_wm = df_no_wm[df_no_wm['length'] == target_length]

    method = method.lower()
    use_z_score_theoretical = True # 是否可以计算理论正态分布阈值
    
    # 根据不同方案读取对应的统计得分列
    # if 'bimark' in method:
    #     score_wm = df_wm['verify_z_score'].dropna().values
    #     score_nowm = df_no_wm['global_z_score'].dropna().values
    # elif 'omnimark' in method:
    #     score_wm = df_wm['global_z_score'].dropna().values
    #     score_nowm = df_no_wm['global_z_score'].dropna().values
    # elif 'watermod' in method:
    #     score_wm = df_wm['z_score'].dropna().values
    #     score_nowm = df_no_wm['global_z_score'].dropna().values
    # elif method in ['xmark', 'stealthink', 'mpac']:
    #     # 这些方法使用比特匹配率作为得分。hit_rate 越高越可能是水印文本。
    #     score_wm = df_wm['hit_rate'].dropna().values
    #     score_nowm = df_no_wm['hit_rate'].dropna().values
    #     use_z_score_theoretical = False 
    # else:
    #     raise ValueError(f"未知的检测方法: {method}")


    if 'bimark' in method:
        score_wm = df_wm['hit_rate'].dropna().values
        score_nowm = df_no_wm['hit_rate'].dropna().values
    elif 'omnimark' in method:
        score_wm = df_wm['hit_rate'].dropna().values
        score_nowm = df_no_wm['hit_rate'].dropna().values
    elif 'watermod' in method:
        score_wm = df_wm['hit_rate'].dropna().values
        score_nowm = df_no_wm['hit_rate'].dropna().values
    elif method in ['xmark', 'stealthink', 'mpac']:
        # 这些方法使用比特匹配率作为得分。hit_rate 越高越可能是水印文本。
        score_wm = df_wm['hit_rate'].dropna().values
        score_nowm = df_no_wm['hit_rate'].dropna().values
        use_z_score_theoretical = False 
    else:
        raise ValueError(f"未知的检测方法: {method}")

    # ================= 1. 经验阈值计算 (ROC) =================
    y_true = np.concatenate([np.ones(len(score_wm)), np.zeros(len(score_nowm))])
    y_scores = np.concatenate([score_wm, score_nowm])

    # 计算 ROC 曲线 (roc_curve 不关心绝对数值，只关心排序单调性)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # 线性插值获取目标 target_fpr 下的 TPR
    empirical_tpr = np.interp(target_fpr, fpr, tpr)

    print(f"\n=== {method.upper()} 零比特水印检出率评估 ===")
    print(f"评估样本数: 有水印 {len(score_wm)} 条, 无水印 {len(score_nowm)} 条")
    print(f"使用的得分指标: {'Z-Score' if use_z_score_theoretical else 'Hit Rate'}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"目标误检率 (FPR): {target_fpr * 100:.2f}%")
    print(f"-> 经验检出率 (TPR@FPR, 基于无水印基线): {empirical_tpr * 100:.2f}%")

    # ================= 2. 理论阈值计算 =================
    if use_z_score_theoretical:
        theoretical_threshold = stats.norm.ppf(1 - target_fpr)
        theoretical_tpr = np.mean(score_wm > theoretical_threshold)
        print(f"-> 理论检出率 (基于 Z>{theoretical_threshold:.3f}): {theoretical_tpr * 100:.2f}%")
    else:
        print(f"-> 理论检出率: (对于 Hit Rate，不适用标准正态理论推导，请以经验检出率为准)")
        
    print("\n")
    return empirical_tpr, roc_auc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wm_csv", type=str, required=True)
    parser.add_argument("--nowm_csv", type=str, required=True)
    parser.add_argument("--method", type=str, default='bimark', help="bimark, omnimark, watermod, xmark, stealthink, mpac")
    parser.add_argument("--fpr", type=float, default=0.01)
    parser.add_argument("--length", type=int, default=None)
    args = parser.parse_args()

    calculate_tpr_at_fpr(args.wm_csv, args.nowm_csv, args.fpr, args.method, args.length)