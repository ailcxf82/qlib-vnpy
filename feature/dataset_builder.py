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
    
    def winsorize_cross_sectional(self, features, quantile_range=[0.01, 0.99]):
        """
        截面缩尾处理：在每个时间点分别进行缩尾
        这样可以保留股票之间的相对差异
        """
        if isinstance(features.index, pd.MultiIndex):
            # 获取时间层级（通常是第二层：datetime）
            datetime_level = features.index.names[1] if len(features.index.names) > 1 else features.index.names[0]
            
            def winsorize_group(x):
                if x.std() < 1e-8:  # 如果标准差太小，跳过
                    return x
                lower = x.quantile(quantile_range[0])
                upper = x.quantile(quantile_range[1])
                return x.clip(lower=lower, upper=upper)
            
            default_logger.info(f"使用截面缩尾处理，按 {datetime_level} 分组")
            winsorized = features.groupby(level=datetime_level).transform(winsorize_group)
            return winsorized
        else:
            # 单时间点数据，直接缩尾
            result = features.copy()
            for col in result.columns:
                lower = result[col].quantile(quantile_range[0])
                upper = result[col].quantile(quantile_range[1])
                result[col] = result[col].clip(lower=lower, upper=upper)
            return result
    
    def standardize_cross_sectional(self, features):
        """
        截面标准化：在每个时间点对所有股票分别标准化
        这是量化投资中的标准做法，可以保留股票之间的相对差异
        """
        if isinstance(features.index, pd.MultiIndex):
            # 获取时间层级
            datetime_level = features.index.names[1] if len(features.index.names) > 1 else features.index.names[0]
            
            default_logger.info(f"使用截面标准化，按 {datetime_level} 分组")
            
            # 按时间分组标准化
            standardized = features.groupby(level=datetime_level).transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
            return standardized
        else:
            # 单时间点数据
            return (features - features.mean()) / (features.std() + 1e-8)
    
    def rank_transform(self, features):
        """
        排名转换：将特征值转换为百分位排名（0-1之间）
        这是最推荐的方法，因为：
        1. 对异常值不敏感
        2. 最大化保留股票之间的相对差异
        3. 所有特征都在相同的尺度上
        
        改进：为相同值添加微小随机扰动，避免大量相同值导致rank后仍然相同
        """
        if isinstance(features.index, pd.MultiIndex):
            # 获取时间层级
            datetime_level = features.index.names[1] if len(features.index.names) > 1 else features.index.names[0]
            
            default_logger.info(f"使用排名转换，按 {datetime_level} 分组")
            
            # 转换为百分位排名
            # 先添加微小随机扰动打破相同值
            np.random.seed(42)  # 固定随机种子保证可重复
            noise = np.random.uniform(-1e-10, 1e-10, features.shape)
            features_noisy = features + noise
            
            ranked = features_noisy.groupby(level=datetime_level).transform(
                lambda x: x.rank(pct=True, method='first')  # 使用'first'方法处理相同值
            )
            return ranked
        else:
            # 单时间点数据
            # 先添加微小随机扰动打破相同值
            np.random.seed(42)  # 固定随机种子保证可重复
            noise = np.random.uniform(-1e-10, 1e-10, features.shape)
            features_noisy = features + noise
            return features_noisy.rank(pct=True, method='first')  # 使用'first'方法处理相同值
    
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
        
        # 填充缺失值（改进版）
        fillna_method = self.feature_engineering_config.get('fillna_method', 'forward')
        default_logger.info(f"使用 {fillna_method} 方法填充缺失值")
        
        if fillna_method == 'forward':
            # 推荐：前向填充，适合技术指标
            features = features.ffill()
            # 如果还有缺失（比如第一行），用后向填充
            features = features.bfill()
        elif fillna_method == 'median_cross_sectional':
            # 截面中位数填充：用同一时间点其他股票的中位数
            if isinstance(features.index, pd.MultiIndex):
                datetime_level = features.index.names[1] if len(features.index.names) > 1 else features.index.names[0]
                features = features.groupby(level=datetime_level).transform(
                    lambda x: x.fillna(x.median())
                )
            else:
                features = features.fillna(features.median())
        elif fillna_method == 'mean':
            # 全局均值填充（不推荐）
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            features[numeric_cols] = features[numeric_cols].fillna(features[numeric_cols].mean())
        elif fillna_method == 'median':
            # 全局中位数填充（不推荐）
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            features[numeric_cols] = features[numeric_cols].fillna(features[numeric_cols].median())
        elif fillna_method == 'backward':
            # 后向填充
            features = features.bfill()
        
        # 缩尾处理（改用截面缩尾）
        if self.feature_engineering_config.get('winsorize', True):
            quantile_range = self.feature_engineering_config.get(
                'winsorize_quantile', [0.01, 0.99]
            )
            # 使用截面缩尾而非全局缩尾
            use_cross_sectional = self.feature_engineering_config.get('winsorize_cross_sectional', True)
            if use_cross_sectional and isinstance(features.index, pd.MultiIndex):
                features = self.winsorize_cross_sectional(features, quantile_range)
                default_logger.info("完成截面缩尾处理")
            else:
                # 全局缩尾（不推荐，仅用于单时间点数据）
                for col in features.columns:
                    if features[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                        features[col] = winsorize(features[col], quantile_range)
                default_logger.info("完成全局缩尾处理")
        
        # 标准化（关键修复：改用截面标准化或排名转换）
        if self.feature_engineering_config.get('standardize', True):
            # 读取标准化方法配置，默认使用排名转换
            standardize_method = self.feature_engineering_config.get('standardize_method', 'rank')
            
            if standardize_method == 'rank':
                # 方法1：排名转换（最推荐）
                features = self.rank_transform(features)
                default_logger.info("✓ 使用排名转换完成特征标准化")
            elif standardize_method == 'cross_sectional':
                # 方法2：截面标准化（备选方案）
                features = self.standardize_cross_sectional(features)
                default_logger.info("✓ 使用截面标准化完成特征标准化")
            elif standardize_method == 'global':
                # 方法3：全局标准化（原方法，不推荐）
                for col in features.columns:
                    if features[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                        features[col] = standardize(features[col])
                default_logger.warning("⚠ 使用全局标准化（不推荐，可能导致预测信号重复）")
            else:
                default_logger.info("跳过特征标准化")
        
        # 再次填充可能产生的NaN（改进：不要简单填充为0）
        # 使用更智能的填充方法，避免产生大量相同的值
        if isinstance(features.index, pd.MultiIndex):
            # MultiIndex：按时间点分组，使用截面中位数填充
            datetime_level = features.index.names[1] if len(features.index.names) > 1 else features.index.names[0]
            features = features.groupby(level=datetime_level).transform(
                lambda x: x.fillna(x.median() if not x.median() == 0 else x.mean())
            )
            # 如果还有NaN（比如所有值都是NaN），使用0填充
            features = features.fillna(0)
        else:
            # 单时间点：使用中位数填充
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                median_val = features[col].median()
                if pd.notna(median_val) and median_val != 0:
                    features[col] = features[col].fillna(median_val)
                else:
                    mean_val = features[col].mean()
                    if pd.notna(mean_val):
                        features[col] = features[col].fillna(mean_val)
                    else:
                        features[col] = features[col].fillna(0)
            # 非数值列用0填充
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

