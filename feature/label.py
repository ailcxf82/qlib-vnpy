"""标签生成模块"""
import pandas as pd
import numpy as np
from utils.logger import default_logger


class LabelGenerator:
    """标签生成器"""
    
    def __init__(self, config):
        self.config = config
    
    def generate_next_week_return(self, data, forward_days=5):
        """
        生成下周收益率标签
        
        Args:
            data: 包含close价格的DataFrame，MultiIndex(instrument, datetime)
            forward_days: 向前看的天数，默认5天（一周）
        
        Returns:
            包含标签的Series
        """
        default_logger.info(f"生成下周收益率标签，向前 {forward_days} 天")
        
        # 确保数据按时间排序
        if isinstance(data.index, pd.MultiIndex):
            data = data.sort_index()
            
            labels = []
            for instrument in data.index.get_level_values(0).unique():
                stock_data = data.loc[instrument]
                if 'close' in stock_data.columns:
                    close = stock_data['close']
                else:
                    close = stock_data.iloc[:, 0]  # 假设第一列是收盘价
                
                # 计算未来收益率
                future_close = close.shift(-forward_days)
                label = (future_close / close) - 1
                
                # 添加instrument信息
                label_df = pd.DataFrame({
                    'label': label,
                    'instrument': instrument
                }, index=stock_data.index)
                
                labels.append(label_df)
            
            all_labels = pd.concat(labels)
            return all_labels
        else:
            # 单个时间序列
            close = data['close'] if 'close' in data.columns else data.iloc[:, 0]
            future_close = close.shift(-forward_days)
            label = (future_close / close) - 1
            return label
    
    def generate_rank_label(self, data, forward_days=5):
        """
        生成排名标签（横截面排名）
        
        Args:
            data: 包含close价格的DataFrame
            forward_days: 向前看的天数
        
        Returns:
            排名标签
        """
        returns = self.generate_next_week_return(data, forward_days)
        
        if isinstance(returns, pd.DataFrame):
            # 按日期分组，计算排名
            returns['date'] = returns.index.get_level_values(1)
            returns['rank'] = returns.groupby('date')['label'].rank(pct=True)
            return returns['rank']
        else:
            return returns.rank(pct=True)
    
    def generate_binary_label(self, data, forward_days=5, threshold=0):
        """
        生成二分类标签（涨跌）
        
        Args:
            data: 包含close价格的DataFrame
            forward_days: 向前看的天数
            threshold: 阈值，默认0
        
        Returns:
            二分类标签（1: 上涨, 0: 下跌）
        """
        returns = self.generate_next_week_return(data, forward_days)
        
        if isinstance(returns, pd.DataFrame):
            return (returns['label'] > threshold).astype(int)
        else:
            return (returns > threshold).astype(int)
    
    def generate_label(self, data, label_type='return', **kwargs):
        """
        生成标签的统一接口
        
        Args:
            data: 数据
            label_type: 标签类型 ('return', 'rank', 'binary')
            **kwargs: 其他参数
        
        Returns:
            标签
        """
        if label_type == 'return':
            return self.generate_next_week_return(data, **kwargs)
        elif label_type == 'rank':
            return self.generate_rank_label(data, **kwargs)
        elif label_type == 'binary':
            return self.generate_binary_label(data, **kwargs)
        else:
            raise ValueError(f"不支持的标签类型: {label_type}")

