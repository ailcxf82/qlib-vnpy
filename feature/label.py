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
                # 支持Qlib的列名（$close）和普通列名（close）
                if '$close' in stock_data.columns:
                    close = stock_data['$close']
                elif 'close' in stock_data.columns:
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
    
    def generate_volatility_adjusted_return(self, data, forward_days=5):
        """
        生成波动率调整后的收益率标签（夏普比率风格）
        这种标签考虑了收益和风险的平衡
        
        Args:
            data: 包含价格数据的DataFrame
            forward_days: 向前看的天数
        
        Returns:
            波动率调整后的收益率
        """
        default_logger.info(f"生成波动率调整收益率标签，向前 {forward_days} 天")
        
        if not isinstance(data.index, pd.MultiIndex):
            return self.generate_next_week_return(data, forward_days)
        
        data = data.sort_index()
        labels = []
        
        for instrument in data.index.get_level_values(0).unique():
            stock_data = data.loc[instrument]
            
            # 获取收盘价
            if '$close' in stock_data.columns:
                close = stock_data['$close']
            elif 'close' in stock_data.columns:
                close = stock_data['close']
            else:
                close = stock_data.iloc[:, 0]
            
            # 计算未来收益率
            future_close = close.shift(-forward_days)
            returns = (future_close / close) - 1
            
            # 计算历史波动率（用过去20天）
            hist_returns = close.pct_change()
            volatility = hist_returns.rolling(window=20, min_periods=5).std()
            
            # 波动率调整（类似信息比率）
            adjusted_returns = returns / (volatility + 1e-4)
            
            label_df = pd.DataFrame({
                'label': adjusted_returns,
                'instrument': instrument
            }, index=stock_data.index)
            
            labels.append(label_df)
        
        all_labels = pd.concat(labels)
        return all_labels
    
    def generate_excess_return(self, data, forward_days=5, benchmark_weight='equal'):
        """
        生成超额收益标签（相对于市场基准）
        这是量化投资中最常用的标签类型
        
        Args:
            data: 包含价格数据的DataFrame
            forward_days: 向前看的天数
            benchmark_weight: 基准权重方式 ('equal': 等权, 'cap': 市值加权)
        
        Returns:
            超额收益标签
        """
        default_logger.info(f"生成超额收益标签，向前 {forward_days} 天")
        
        if not isinstance(data.index, pd.MultiIndex):
            return self.generate_next_week_return(data, forward_days)
        
        data = data.sort_index()
        
        # 先计算所有股票的收益率
        all_returns_dict = {}  # {date: {instrument: return}}
        
        for instrument in data.index.get_level_values(0).unique():
            stock_data = data.loc[instrument]
            
            # 支持Qlib的列名（$close）和普通列名（close）
            if '$close' in stock_data.columns:
                close = stock_data['$close']
            elif 'close' in stock_data.columns:
                close = stock_data['close']
            else:
                close = stock_data.iloc[:, 0]
            
            future_close = close.shift(-forward_days)
            returns = (future_close / close) - 1
            
            # 按日期组织
            for date, ret in returns.items():
                if date not in all_returns_dict:
                    all_returns_dict[date] = {}
                all_returns_dict[date][instrument] = ret
        
        # 计算超额收益并构建标签（与原方法格式一致）
        labels = []
        for instrument in data.index.get_level_values(0).unique():
            stock_data = data.loc[instrument]
            
            excess_returns = []
            for date in stock_data.index:
                if date in all_returns_dict:
                    # 计算市场平均收益
                    market_return = np.nanmean(list(all_returns_dict[date].values()))
                    # 计算超额收益
                    stock_return = all_returns_dict[date].get(instrument, np.nan)
                    excess_return = stock_return - market_return if not np.isnan(stock_return) and not np.isnan(market_return) else np.nan
                    excess_returns.append(excess_return)
                else:
                    excess_returns.append(np.nan)
            
            # 创建标签DataFrame（与原方法格式一致）
            label_df = pd.DataFrame({
                'label': excess_returns,
                'instrument': instrument
            }, index=stock_data.index)
            
            labels.append(label_df)
        
        all_labels = pd.concat(labels)
        return all_labels
    
    def generate_quantile_label(self, data, forward_days=5, n_quantiles=5):
        """
        生成分位数标签（多分类）
        将股票按未来收益率分成N个档次
        
        Args:
            data: 包含价格数据的DataFrame
            forward_days: 向前看的天数
            n_quantiles: 分位数数量（默认5档）
        
        Returns:
            分位数标签 (0, 1, 2, ..., n_quantiles-1)
        """
        default_logger.info(f"生成 {n_quantiles} 档分位数标签")
        
        # 先生成收益率
        returns = self.generate_next_week_return(data, forward_days)
        
        if isinstance(returns, pd.DataFrame):
            # 按日期分组，计算分位数
            if 'label' in returns.columns:
                returns_series = returns['label']
            else:
                returns_series = returns.iloc[:, 0]
            
            # 获取日期索引
            if isinstance(returns_series.index, pd.MultiIndex):
                dates = returns_series.index.get_level_values(1)
            else:
                dates = returns_series.index
            
            # 按日期分组，计算分位数标签
            quantile_labels = returns_series.groupby(dates).transform(
                lambda x: pd.qcut(x, n_quantiles, labels=False, duplicates='drop')
            )
            
            return pd.DataFrame({'label': quantile_labels}, index=returns_series.index)
        else:
            return pd.qcut(returns, n_quantiles, labels=False, duplicates='drop')
    
    def generate_label(self, data, label_type='return', **kwargs):
        """
        生成标签的统一接口
        
        Args:
            data: 数据
            label_type: 标签类型
                - 'return': 简单收益率（默认）
                - 'excess_return': 超额收益率（推荐）⭐
                - 'volatility_adjusted': 波动率调整收益率
                - 'rank': 排名标签
                - 'binary': 二分类标签
                - 'quantile': 分位数标签
            **kwargs: 其他参数
        
        Returns:
            标签
        """
        if label_type == 'return':
            return self.generate_next_week_return(data, **kwargs)
        elif label_type == 'excess_return':
            return self.generate_excess_return(data, **kwargs)
        elif label_type == 'volatility_adjusted':
            return self.generate_volatility_adjusted_return(data, **kwargs)
        elif label_type == 'rank':
            return self.generate_rank_label(data, **kwargs)
        elif label_type == 'binary':
            return self.generate_binary_label(data, **kwargs)
        elif label_type == 'quantile':
            return self.generate_quantile_label(data, **kwargs)
        else:
            raise ValueError(f"不支持的标签类型: {label_type}")

