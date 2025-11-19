"""模型预测器"""
import pandas as pd
import numpy as np
from model.trainer import ModelTrainer
from utils.logger import default_logger
from utils.timer import timing_decorator
from utils.tools import ensure_dir


class ModelPredictor:
    """模型预测器"""
    
    def __init__(self, model_config):
        self.model_config = model_config
        self.trainer = ModelTrainer(model_config)
    
    def load_model(self, model_path):
        """加载模型"""
        self.trainer.load_model(model_path)
    
    @timing_decorator
    def predict(self, features, date=None):
        """
        预测
        
        Args:
            features: 特征数据（DataFrame）
            date: 预测日期（可选）
        
        Returns:
            预测结果DataFrame，columns: [instrument, score]
        """
        default_logger.info(f"开始预测，样本数: {len(features)}")
        
        # 提取instrument信息
        if 'instrument' in features.columns:
            instruments = features['instrument'].values
            features = features.drop('instrument', axis=1)
        elif isinstance(features.index, pd.MultiIndex):
            instruments = features.index.get_level_values(0).values
        else:
            instruments = np.arange(len(features))
        
        # 预测
        scores = self.trainer.predict(features)
        
        # 构建结果DataFrame
        result = pd.DataFrame({
            'instrument': instruments,
            'score': scores
        })
        
        if date is not None:
            result['date'] = date
        
        default_logger.info("预测完成")
        
        return result
    
    def predict_and_save(self, features, output_path, date=None):
        """
        预测并保存结果
        
        Args:
            features: 特征数据
            output_path: 输出路径
            date: 预测日期
        """
        # 预测
        result = self.predict(features, date)
        
        # 保存
        ensure_dir(output_path)
        result.to_csv(output_path, index=False)
        
        default_logger.info(f"预测结果已保存: {output_path}")
        
        return result
    
    def batch_predict(self, features_dict, output_dir):
        """
        批量预测
        
        Args:
            features_dict: 特征字典，{date: features}
            output_dir: 输出目录
        
        Returns:
            预测结果字典
        """
        ensure_dir(output_dir)
        
        results = {}
        
        for date, features in features_dict.items():
            output_path = f"{output_dir}/pred_{date}.csv"
            result = self.predict_and_save(features, output_path, date)
            results[date] = result
        
        default_logger.info(f"批量预测完成，共 {len(results)} 个日期")
        
        return results

