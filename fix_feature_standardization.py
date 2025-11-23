"""修复特征标准化问题 - 使用截面标准化"""
import pandas as pd
import numpy as np

def standardize_cross_sectional(features):
    """
    截面标准化：在每个时间点对所有股票进行标准化
    这样可以保留股票之间的相对差异
    """
    if isinstance(features.index, pd.MultiIndex):
        # 多层索引：按时间分组标准化
        level_name = features.index.names[1]  # datetime level
        
        standardized = features.groupby(level=level_name).transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
        return standardized
    else:
        # 单层索引：直接标准化
        return (features - features.mean()) / (features.std() + 1e-8)


def rank_transform(features):
    """
    排名转换：将特征值转换为排名（0-1之间）
    这是最有效的特征转换方式，能最大化保留股票间的相对差异
    """
    if isinstance(features.index, pd.MultiIndex):
        level_name = features.index.names[1]
        
        ranked = features.groupby(level=level_name).transform(
            lambda x: x.rank(pct=True)  # 百分位排名
        )
        return ranked
    else:
        return features.rank(pct=True)


def winsorize_cross_sectional(features, quantile_range=[0.01, 0.99]):
    """
    截面缩尾：在每个时间点分别进行缩尾处理
    """
    if isinstance(features.index, pd.MultiIndex):
        level_name = features.index.names[1]
        
        def winsorize_group(x):
            lower = x.quantile(quantile_range[0])
            upper = x.quantile(quantile_range[1])
            return x.clip(lower=lower, upper=upper)
        
        winsorized = features.groupby(level=level_name).transform(winsorize_group)
        return winsorized
    else:
        lower = features.quantile(quantile_range[0])
        upper = features.quantile(quantile_range[1])
        return features.clip(lower=lower, upper=upper, axis=1)


# 示例：如何使用
if __name__ == "__main__":
    print("特征预处理修复方案")
    print("="*60)
    
    print("\n推荐的特征预处理流程：\n")
    print("1. 填充缺失值（使用前向填充或行业中位数）")
    print("2. 截面缩尾处理（winsorize_cross_sectional）")
    print("3. 排名转换（rank_transform）← 最推荐")
    print("   或")
    print("   截面标准化（standardize_cross_sectional）")
    
    print("\n" + "="*60)
    print("修改建议：")
    print("="*60)
    print("\n在 feature/dataset_builder.py 的 preprocess_features 方法中：\n")
    
    print("# 原来的代码（有问题）：")
    print("features[col] = standardize(features[col])  # ← 全局标准化\n")
    
    print("# 修改为（推荐方案1 - 排名转换）：")
    print("features = rank_transform(features)  # ← 截面排名\n")
    
    print("# 或修改为（方案2 - 截面标准化）：")
    print("features = standardize_cross_sectional(features)  # ← 截面标准化\n")
    
    print("\n这样可以最大程度保留不同股票之间的差异！")





