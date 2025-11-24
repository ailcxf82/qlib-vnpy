"""滚动训练主程序"""
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)  # 切换工作目录到项目根目录

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
#from feature.feature_pipeline import FeaturePipeline
from feature.qlib_feature_pipeline import QlibFeaturePipeline
from model.trainer import ModelTrainer
from utils.logger import default_logger, setup_logger
from utils.timer import Timer
from utils.tools import load_yaml, ensure_dir, get_trade_dates


class RollingTrainer:
    """滚动训练器"""
    
    def __init__(self, data_config, pipeline_config, model_config):
        self.data_config = data_config
        self.pipeline_config = pipeline_config
        self.model_config = model_config
        
        # self.feature_pipeline = FeaturePipeline(data_config, pipeline_config)
        self.qlib_feature_pipeline = QlibFeaturePipeline(data_config, pipeline_config)
        self.rolling_config = pipeline_config.get('rolling', {})
        
        self.train_window = self.rolling_config.get('train_window', 156)  # 3年
        self.step = self.rolling_config.get('step', 1)  # 每周滚动1周
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
        # 获取周频交易日期
        # all_weeks = get_trade_dates(start_date, end_date, freq='W-MON')
        
        # 按交易日
        from utils.tools import get_real_trade_dates
        all_weeks = get_real_trade_dates(start_date, end_date)
        
        windows = []
        if self.is_weekly == False:
            for i in range(self.train_window, len(all_weeks)):
                train_start = all_weeks[i - self.train_window]
                train_end = all_weeks[i - 1]
                test_date = all_weeks[i] if i < len(all_weeks) else all_weeks[-1]
                windows.append((train_start, train_end, test_date))

            default_logger.info(f"总共 {len(windows)} 个滚动窗口")
            return windows
        
        else:
            for i in range(self.train_window, len(all_weeks), self.step):
                train_start = all_weeks[i - self.train_window]
                train_end = all_weeks[i - 1]
                test_date = all_weeks[i] if i < len(all_weeks) else all_weeks[-1]
                
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
        
        # 1. 生成特征和标签
        with Timer("特征生成"):
            # features, labels = self.feature_pipeline.run(
            #     start_time=train_start,
            #     end_time=train_end,
            #     instruments='csi300_file'  # 从文件读取CSI300股票代码
            # )
            features, labels = self.qlib_feature_pipeline.run(
                start_time=train_start,
                end_time=train_end, 
                instruments='csi300_file'
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
        
        return trainer
    
    def run(self, start_date, end_date, save_dir='data'):
        """
        运行滚动训练
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            save_dir: 保存目录
        """
        default_logger.info("开始滚动训练流程")
        default_logger.info(f"时间范围: {start_date} ~ {end_date}")
        default_logger.info(f"训练窗口: {self.train_window} 周")
        default_logger.info(f"滚动步长: {self.step} 周")
        
        # 获取滚动窗口
        windows = self.get_rolling_windows(start_date, end_date)
        
        # 训练每个窗口
        trained_models = []
        
        for train_start, train_end, test_date in windows:
            try:
                trainer = self.train_single_window(
                    train_start, train_end, test_date, save_dir
                )
                
                if trainer is not None:
                    trained_models.append({
                        'test_date': test_date, 
                        'trainer': trainer
                    })
            except Exception as e:
                default_logger.error(f"训练窗口 {test_date} 失败: {e}")
                continue
        
        default_logger.info("=" * 80)
        default_logger.info(f"滚动训练完成，成功训练 {len(trained_models)}/{len(windows)} 个模型")
        default_logger.info("=" * 80)
        
        return trained_models


def main():
    """主函数"""
    # 设置日志
    logger = setup_logger(
        'rolling_train',
        f'logs/rolling_train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    
    # 加载配置
    data_config = load_yaml('config/data.yaml')
    pipeline_config = load_yaml('config/pipeline.yaml')
    model_config = load_yaml('config/model.yaml')
    
    # 创建滚动训练器
    trainer = RollingTrainer(data_config, pipeline_config, model_config)
    
    # 运行训练
    training_config = pipeline_config.get('training', {})
    start_date = training_config.get('start_date', '2017-01-01')
    end_date = training_config.get('end_date', '2024-12-31')
    
    with Timer("总训练时间"):
        trainer.run(start_date, end_date)
    
    logger.info("滚动训练流程结束")


if __name__ == '__main__':
    main()

