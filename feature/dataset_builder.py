"""数据集构建模块"""
import pandas as pd
import numpy as np
from utils.logger import default_logger
from utils.tools import winsorize, standardize


class DatasetBuilder:
    """数据集构建器"""
    
    def __init__(self, config):
        self.config = config
        self.feature_engineering_config = config.get('feature_engineering', {})
    
    def align_features_labels(self, features, labels):
        """对齐特征和标签"""
        if isinstance(features, pd.DataFrame) and isinstance(labels, pd.DataFrame):
            # 确保索引对齐
            common_index = features.index.intersection(labels.index)
            features = features.loc[common_index]
            labels = labels.loc[common_index]
            
            default_logger.info(f"对齐后的数据量: {len(common_index)}")
            return features, labels
        else:
            return features, labels
    
    def preprocess_features(self, features):
        """预处理特征"""
        default_logger.info("开始特征预处理")
        
        # 确保没有非数值列（如 instrument 列）
        if 'instrument' in features.columns:
            default_logger.warning("检测到 'instrument' 列，将在预处理前移除")
            instrument_col = features['instrument']
            features = features.drop('instrument', axis=1)
            has_instrument = True
        else:
            has_instrument = False
        
        # 填充缺失值（只对数值列）
        fillna_method = self.feature_engineering_config.get('fillna_method', 'mean')
        default_logger.info(f"使用 {fillna_method} 方法填充缺失值")
        
        if fillna_method == 'mean':
            # 只对数值列计算均值
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            features[numeric_cols] = features[numeric_cols].fillna(features[numeric_cols].mean())
        elif fillna_method == 'median':
            # 只对数值列计算中位数
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            features[numeric_cols] = features[numeric_cols].fillna(features[numeric_cols].median())
        elif fillna_method == 'forward':
            # 前向填充（使用新的API）
            features = features.ffill()
        elif fillna_method == 'backward':
            # 后向填充（使用新的API）
            features = features.bfill()
        
        # 缩尾处理
        if self.feature_engineering_config.get('winsorize', True):
            quantile_range = self.feature_engineering_config.get(
                'winsorize_quantile', [0.01, 0.99]
            )
            for col in features.columns:
                if features[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    features[col] = winsorize(features[col], quantile_range)
        
        # 标准化
        if self.feature_engineering_config.get('standardize', True):
            for col in features.columns:
                if features[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    features[col] = standardize(features[col])
        
        # 再次填充可能产生的NaN
        features = features.fillna(0)
        
        # 替换inf
        features = features.replace([np.inf, -np.inf], 0)
        
        # 恢复 instrument 列（如果之前移除了）
        if has_instrument:
            features['instrument'] = instrument_col
            default_logger.info("已恢复 'instrument' 列")
        
        default_logger.info(f"特征预处理完成: {features.shape}")
        return features
    
    def split_train_test(self, features, labels, split_date=None, split_ratio=0.8):
        """
        划分训练集和测试集
        
        Args:
            features: 特征数据
            labels: 标签数据
            split_date: 划分日期（如果指定，按日期划分）
            split_ratio: 划分比例（如果不指定日期，按比例划分）
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        if isinstance(features.index, pd.MultiIndex):
            # 多层索引，按日期划分
            dates = features.index.get_level_values(1).unique().sort_values()
            
            if split_date is None:
                split_idx = int(len(dates) * split_ratio)
                split_date = dates[split_idx]
            
            train_mask = features.index.get_level_values(1) < split_date
            test_mask = features.index.get_level_values(1) >= split_date
            
            X_train = features[train_mask]
            X_test = features[test_mask]
            y_train = labels[train_mask]
            y_test = labels[test_mask]
        else:
            # 单层索引，按比例划分
            split_idx = int(len(features) * split_ratio)
            X_train = features.iloc[:split_idx]
            X_test = features.iloc[split_idx:]
            y_train = labels.iloc[:split_idx]
            y_test = labels.iloc[split_idx:]
        
        default_logger.info(
            f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}"
        )
        
        return X_train, X_test, y_train, y_test
    
    def build_dataset(self, features, labels, preprocess=True):
        """
        构建完整的数据集
        
        Args:
            features: 特征数据
            labels: 标签数据
            preprocess: 是否预处理
        
        Returns:
            处理后的特征和标签
        """
        # 对齐
        features, labels = self.align_features_labels(features, labels)
        
        # 预处理
        if preprocess:
            features = self.preprocess_features(features)
        
        # 移除标签为NaN的样本
        if isinstance(labels, pd.DataFrame):
            valid_mask = labels['label'].notna()
        else:
            valid_mask = labels.notna()
        
        features = features[valid_mask]
        labels = labels[valid_mask]
        
        default_logger.info(f"最终数据集大小: {len(features)}")
        
        return features, labels

