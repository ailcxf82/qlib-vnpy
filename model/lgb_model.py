"""LightGBM模型封装"""
import lightgbm as lgb
import pandas as pd
import numpy as np
from utils.logger import default_logger


class LGBModel:
    """LightGBM模型包装类"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.feature_names = None
        self.feature_importance = None
    
    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_valid: 验证特征
            y_valid: 验证标签
        
        Returns:
            训练好的模型
        """
        default_logger.info("开始训练LightGBM模型")
        
        # 保存特征名称
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.values.ravel()
        elif isinstance(y_train, pd.Series):
            y_train = y_train.values
        
        # 创建数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_valid is not None and y_valid is not None:
            if isinstance(X_valid, pd.DataFrame):
                X_valid = X_valid.values
            if isinstance(y_valid, pd.DataFrame):
                y_valid = y_valid.values.ravel()
            elif isinstance(y_valid, pd.Series):
                y_valid = y_valid.values
            
            valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
            valid_sets.append(valid_data)
            valid_names.append('valid')
        
        # 训练参数
        params = self.config.copy()
        num_boost_round = params.pop('n_estimators', 100)
        
        # 训练
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[lgb.log_evaluation(period=50)]
        )
        
        # 特征重要度
        self.feature_importance = self.model.feature_importance(importance_type='gain')
        
        default_logger.info("LightGBM模型训练完成")
        
        return self.model
    
    def predict(self, X):
        """
        预测
        
        Args:
            X: 特征数据
        
        Returns:
            预测结果
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train方法")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        pred = self.model.predict(X)
        
        return pred
    
    def get_feature_importance(self, top_n=20):
        """
        获取特征重要度
        
        Args:
            top_n: 返回top N个特征
        
        Returns:
            特征重要度DataFrame
        """
        if self.feature_importance is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importance
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def print_feature_importance(self, top_n=20):
        """打印特征重要度"""
        importance_df = self.get_feature_importance(top_n)
        
        if importance_df is not None:
            default_logger.info("=" * 60)
            default_logger.info(f"Top {top_n} 特征重要度:")
            for idx, row in importance_df.iterrows():
                default_logger.info(f"  {row['feature']}: {row['importance']:.2f}")
            default_logger.info("=" * 60)
    
    def save_model(self, model_path):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        self.model.save_model(model_path)
        default_logger.info(f"模型已保存: {model_path}")
    
    def load_model(self, model_path):
        """加载模型"""
        self.model = lgb.Booster(model_file=model_path)
        default_logger.info(f"模型已加载: {model_path}")
        
        return self.model

