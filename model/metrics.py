"""模型评估指标"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from utils.logger import default_logger


class Metrics:
    """评估指标"""
    
    @staticmethod
    def calculate_ic(pred, label):
        """计算信息系数 (IC)"""
        if len(pred) < 2:
            return np.nan
        
        # 移除NaN
        mask = (~np.isnan(pred)) & (~np.isnan(label))
        pred = pred[mask]
        label = label[mask]
        
        if len(pred) < 2:
            return np.nan
        
        ic, _ = pearsonr(pred, label)
        return ic
    
    @staticmethod
    def calculate_rank_ic(pred, label):
        """计算排名信息系数 (Rank IC)"""
        if len(pred) < 2:
            return np.nan
        
        # 移除NaN
        mask = (~np.isnan(pred)) & (~np.isnan(label))
        pred = pred[mask]
        label = label[mask]
        
        if len(pred) < 2:
            return np.nan
        
        rank_ic, _ = spearmanr(pred, label)
        return rank_ic
    
    @staticmethod
    def calculate_mse(pred, label):
        """计算均方误差"""
        mask = (~np.isnan(pred)) & (~np.isnan(label))
        return np.mean((pred[mask] - label[mask]) ** 2)
    
    @staticmethod
    def calculate_mae(pred, label):
        """计算平均绝对误差"""
        mask = (~np.isnan(pred)) & (~np.isnan(label))
        return np.mean(np.abs(pred[mask] - label[mask]))
    
    @staticmethod
    def calculate_long_short_return(pred, label, top_n=20, bottom_n=20):
        """
        计算多空组合收益
        
        Args:
            pred: 预测值
            label: 真实标签（收益率）
            top_n: 做多的股票数量
            bottom_n: 做空的股票数量
        
        Returns:
            多空组合收益
        """
        if len(pred) < (top_n + bottom_n):
            return np.nan
        
        # 按预测值排序
        df = pd.DataFrame({'pred': pred, 'label': label})
        df = df.dropna()
        df = df.sort_values('pred', ascending=False)
        
        # 多头组合
        long_return = df.head(top_n)['label'].mean()
        
        # 空头组合
        short_return = df.tail(bottom_n)['label'].mean()
        
        # 多空收益
        long_short_return = long_return - short_return
        
        return long_short_return
    
    @staticmethod
    def evaluate_model(pred, label):
        """
        综合评估模型
        
        Returns:
            评估指标字典
        """
        metrics = {
            'IC': Metrics.calculate_ic(pred, label),
            'Rank_IC': Metrics.calculate_rank_ic(pred, label),
            'MSE': Metrics.calculate_mse(pred, label),
            'MAE': Metrics.calculate_mae(pred, label),
            'Long_Short_Return': Metrics.calculate_long_short_return(pred, label)
        }
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics):
        """打印评估指标"""
        default_logger.info("=" * 50)
        default_logger.info("模型评估指标:")
        for key, value in metrics.items():
            default_logger.info(f"  {key}: {value:.4f}")
        default_logger.info("=" * 50)

