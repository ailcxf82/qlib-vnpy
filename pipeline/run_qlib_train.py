"""使用 qlib 标准方式训练模型"""
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import pandas as pd
import numpy as np
from datetime import datetime
from feature.qlib_feature_pipeline import QlibFeaturePipeline
from model.trainer import ModelTrainer
from utils.logger import default_logger, setup_logger
from utils.timer import Timer
from utils.tools import load_yaml, ensure_dir, get_real_trade_dates


class QlibRollingTrainer:
    """使用 qlib 标准方式的滚动训练器"""
    
    def __init__(self, data_config, pipeline_config, model_config):
        self.data_config = data_config
        self.pipeline_config = pipeline_config
        self.model_config = model_config
        
        # 使用新的 qlib feature pipeline
        self.feature_pipeline = QlibFeaturePipeline(data_config, pipeline_config)
        self.rolling_config = pipeline_config.get('rolling', {})
        
        self.train_window = self.rolling_config.get('train_window', 156)  # 3年
        self.step = self.rolling_config.get('step', 1)
        self.training_backend = model_config.get('training_backend', 'native').lower()
        self.use_qlib_backend = self.training_backend == 'qlib'
        self.is_weekly = self.rolling_config.get('is_weekly', False)
    
    def get_rolling_windows(self, start_date, end_date):
        """
        获取滚动窗口
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            窗口列表: [(train_start, train_end, test_date), ...]
        """
        # 获取交易日列表
        all_dates = get_real_trade_dates(start_date, end_date)
        
        windows = []
        if self.is_weekly == False:
            for i in range(self.train_window, len(all_dates)):
                train_start = all_dates[i - self.train_window]
                train_end = all_dates[i - 1]
                test_date = all_dates[i] if i < len(all_dates) else all_dates[-1]
                windows.append((train_start, train_end, test_date))
        else:
            for i in range(self.train_window, len(all_dates), self.step):
                train_start = all_dates[i - self.train_window]
                train_end = all_dates[i - 1]
                test_date = all_dates[i] if i < len(all_dates) else all_dates[-1]
                windows.append((train_start, train_end, test_date))
        
        default_logger.info(f"总共 {len(windows)} 个滚动窗口")
        return windows
    
    def train_single_window(self, train_start, train_end, test_date, save_dir):
        """
        训练单个窗口
        
        Args:
            train_start: 训练开始日期
            train_end: 训练结束日期
            test_date: 测试日期
            save_dir: 保存目录
        
        Returns:
            训练好的模型
        """
        default_logger.info("=" * 80)
        default_logger.info(f"训练窗口: {train_start} ~ {train_end}, 测试日期: {test_date}")
        default_logger.info("=" * 80)
        
        try:
            # 1. 生成特征和标签（使用 qlib 标准方式）
            with Timer("特征生成"):
                if self.use_qlib_backend:
                    # 使用 qlib Dataset 对象
                    dataset = self.feature_pipeline.run(
                        start_time=train_start,
                        end_time=train_end,
                        instruments='csi300_file',
                        return_dataset=True
                    )
                    # 从 dataset 中提取特征和标签
                    handler = dataset.handler
                    features = handler.fetch(col_set="feature", data_key="infer")
                    labels = handler.fetch(col_set="label", data_key="infer")
                    
                    # 转换为 DataFrame 格式
                    if isinstance(labels, pd.DataFrame):
                        if len(labels.columns) == 1:
                            labels = labels.iloc[:, 0]
                        else:
                            labels = labels.iloc[:, 0]
                    if isinstance(labels, pd.Series):
                        labels = pd.DataFrame({'label': labels})
                else:
                    # 使用 DataFrame 格式（兼容原有接口）
                    features, labels = self.feature_pipeline.run(
                        start_time=train_start,
                        end_time=train_end,
                        instruments='csi300_file',
                        return_dataset=False
                    )
            
            if features is None or len(features) == 0:
                default_logger.warning("特征生成失败，跳过此窗口")
                return None
            
            # 2. 准备训练数据
            if isinstance(labels, pd.DataFrame):
                y = labels['label'].values
            else:
                y = labels.values
            
            # 移除包含instrument的列
            if 'instrument' in features.columns:
                X = features.drop('instrument', axis=1)
            else:
                X = features
            
            # 3. 训练模型
            trainer = ModelTrainer(self.model_config, self.pipeline_config)
            
            with Timer("模型训练"):
                if self.use_qlib_backend:
                    trainer.train(features, labels)
                else:
                    trainer.train(X, y)
            
            # 4. 保存模型
            model_dir = f"{save_dir}/models"
            ensure_dir(model_dir)
            model_path = f"{model_dir}/model_{test_date.replace('-', '')}.txt"
            trainer.save_model(model_path)
            
            default_logger.info(f"模型已保存到: {model_path}")
            return trainer
            
        except Exception as e:
            default_logger.error(f"训练窗口 {test_date} 失败: {e}", exc_info=True)
            return None
    
    def run(self, start_date, end_date, save_dir='data'):
        """
        运行滚动训练
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            save_dir: 保存目录
        """
        default_logger.info("开始滚动训练流程（使用 qlib 标准方式）")
        default_logger.info(f"时间范围: {start_date} ~ {end_date}")
        default_logger.info(f"训练窗口: {self.train_window} 天")
        default_logger.info(f"滚动步长: {self.step} 天")
        
        # 获取滚动窗口
        windows = self.get_rolling_windows(start_date, end_date)
        
        # 训练每个窗口
        trained_models = []
        
        for train_start, train_end, test_date in windows:
            trainer = self.train_single_window(
                train_start, train_end, test_date, save_dir
            )
            
            if trainer is not None:
                trained_models.append({
                    'test_date': test_date, 
                    'trainer': trainer
                })
        
        default_logger.info("=" * 80)
        default_logger.info(f"滚动训练完成，成功训练 {len(trained_models)}/{len(windows)} 个模型")
        default_logger.info("=" * 80)
        
        return trained_models


def main():
    """主函数"""
    # 设置日志
    logger = setup_logger(
        'qlib_rolling_train',
        f'logs/qlib_rolling_train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    
    # 加载配置
    data_config = load_yaml('config/data.yaml')
    pipeline_config = load_yaml('config/pipeline.yaml')
    model_config = load_yaml('config/model.yaml')
    
    # 创建滚动训练器
    trainer = QlibRollingTrainer(data_config, pipeline_config, model_config)
    
    # 运行训练
    training_config = pipeline_config.get('training', {})
    start_date = training_config.get('start_date', '2017-01-01')
    end_date = training_config.get('end_date', '2024-12-31')
    
    with Timer("总训练时间"):
        trainer.run(start_date, end_date)
    
    logger.info("滚动训练流程结束")


if __name__ == '__main__':
    main()


