"""滚动预测主程序"""
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)  # 切换工作目录到项目根目录

import pandas as pd
import numpy as np
from datetime import datetime
#from feature.feature_pipeline import FeaturePipeline
from feature.qlib_feature_pipeline import QlibFeaturePipeline
from model.predictor import ModelPredictor
from utils.logger import default_logger, setup_logger
from utils.timer import Timer
from utils.tools import load_yaml, ensure_dir, get_trade_dates


class RollingPredictor:
    """滚动预测器"""
    
    def __init__(self, data_config, pipeline_config, model_config):
        self.data_config = data_config
        self.pipeline_config = pipeline_config
        self.model_config = model_config
        
        # self.feature_pipeline = FeaturePipeline(data_config, pipeline_config)
        self.qlib_feature_pipeline = QlibFeaturePipeline(data_config, pipeline_config)
        self.predictor = ModelPredictor(model_config)
    
    def predict_single_date(self, predict_date, model_path, save_dir):
        """
        预测单个日期
        
        Args:
            predict_date: 预测日期
            model_path: 模型路径
            save_dir: 保存目录
        
        Returns:
            预测结果
        """
        default_logger.info("=" * 80)
        default_logger.info(f"预测日期: {predict_date}")
        default_logger.info(f"模型路径: {model_path}")
        default_logger.info("=" * 80)
        
        # 1. 加载模型
        self.predictor.load_model(model_path)
        
        # 2. 生成特征（使用截止到预测日期前的数据）
        # 计算特征需要历史数据，这里使用过去60天的数据来计算因子
        from datetime import timedelta
        feature_start = (pd.to_datetime(predict_date) - timedelta(days=120)).strftime('%Y-%m-%d')
        
        # 应该使用前一个交易日的数据
        from utils.tools import get_real_trade_dates
        trade_dates = get_real_trade_dates(feature_start, predict_date)
        if len(trade_dates) > 1:
            end_time = trade_dates[-2]  # 前一个交易日
        else:
            end_time = feature_start

        with Timer("特征生成"):
            # features, _ = self.feature_pipeline.run(
            #     start_time=feature_start,
            #     end_time=end_time,
            #     instruments='csi300_file'  # 从文件读取CSI300股票代码
            # )
            features, _ = self.qlib_feature_pipeline.run(
                start_time=feature_start,
                end_time=end_time, 
                instruments='csi300_file'
            )
        
        if features is None or len(features) == 0:
            default_logger.warning("特征生成失败")
            return None
        
        # 3. 提取最新日期的特征 - 修复版
        default_logger.info(f"特征数据形状: {features.shape}")
        default_logger.info(f"特征索引类型: {type(features.index)}")
        
        if isinstance(features.index, pd.MultiIndex):
            # 情况1: MultiIndex (instrument, datetime) 或 (datetime, instrument)
            # 判断哪一层是日期
            index_names = features.index.names
            
            if 'datetime' in index_names:
                date_level = 'datetime'
            elif 'date' in index_names:
                date_level = 'date'
            else:
                # 假设第二层是日期
                date_level = 1
            
            latest_date = features.index.get_level_values(date_level).max()
            latest_features = features.xs(latest_date, level=date_level)
            
            default_logger.info(f"从 MultiIndex 提取最新日期: {latest_date}")
            
        elif 'instrument' in features.columns:
            # 情况2: 单层索引是日期，instrument 是列（实际情况）
            # 获取最新日期
            latest_date = features.index.max()
            latest_features = features[features.index == latest_date].copy()
            
            # 设置 instrument 为索引（方便后续处理）
            if len(latest_features) > 0:
                latest_features = latest_features.reset_index(drop=True)
            
            default_logger.info(f"从单层索引提取最新日期: {latest_date}")
            default_logger.info(f"最新日期的股票数: {len(latest_features)}")
            
        else:
            # 情况3: 已经是单日期的数据
            latest_features = features
            default_logger.warning("无法确定日期结构，使用全部数据")
        
        # 验证数据量
        expected_stocks = 300  # CSI300 大约有 300 只股票
        actual_stocks = len(latest_features)
        
        if actual_stocks > expected_stocks * 2:
            default_logger.error(
                f"❌ 数据异常：预期约 {expected_stocks} 只股票，实际 {actual_stocks} 行\n"
                f"   这可能包含了多个日期的数据！"
            )
            # 强制只取最后 expected_stocks 行（按日期排序后）
            if isinstance(latest_features.index, pd.DatetimeIndex):
                latest_features = latest_features.sort_index().tail(expected_stocks)
            else:
                latest_features = latest_features.tail(expected_stocks)
            default_logger.warning(f"强制截取最后 {expected_stocks} 行")
        
        default_logger.info(f"最终用于预测的样本数: {len(latest_features)}")
        
        # 4. 预测
        output_dir = f"{save_dir}/predictions"
        ensure_dir(output_dir)
        output_path = f"{output_dir}/pred_{predict_date}.csv"
        
        with Timer("模型预测"):
            result = self.predictor.predict_and_save(
                latest_features, output_path, predict_date
            )
        
        default_logger.info(f"预测完成，共 {len(result)} 只股票")
        
        return result
    
    def run(self, start_date, end_date, model_dir='data/models', save_dir='data'):
        """
        运行滚动预测
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            model_dir: 模型目录
            save_dir: 保存目录
        """
        default_logger.info("开始滚动预测流程")
        default_logger.info(f"时间范围: {start_date} ~ {end_date}")
        
        # 获取所有周一日期
        # predict_dates = get_trade_dates(start_date, end_date, freq='W-MON')
        
        # 按交易日
        from utils.tools import get_real_trade_dates
        predict_dates = get_real_trade_dates(start_date, end_date)

        # 为每个日期预测
        predictions = [] 
        
        for predict_date in predict_dates:
            # 查找对应的模型
            model_name = f"model_{predict_date.replace('-', '')}.txt"
            model_path = f"{model_dir}/{model_name}"
            
            if not os.path.exists(model_path):
                default_logger.warning(f"模型不存在: {model_path}，跳过")
                continue
            
            try:
                result = self.predict_single_date(predict_date, model_path, save_dir)
                
                if result is not None:
                    predictions.append({
                        'date': predict_date,
                        'result': result
                    })
            except Exception as e:
                default_logger.error(f"预测日期 {predict_date} 失败: {e}")
                continue
        
        default_logger.info("=" * 80)
        default_logger.info(f"滚动预测完成，成功预测 {len(predictions)}/{len(predict_dates)} 个日期")
        default_logger.info("=" * 80)
        
        return predictions


def main():
    """主函数"""
    # 设置日志
    logger = setup_logger(
        'rolling_predict',
        f'logs/rolling_predict_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    
    # 加载配置
    data_config = load_yaml('config/data.yaml')
    pipeline_config = load_yaml('config/pipeline.yaml')
    model_config = load_yaml('config/model.yaml')
    
    # 创建滚动预测器
    predictor = RollingPredictor(data_config, pipeline_config, model_config)
    
    # 运行预测
    training_config = pipeline_config.get('training', {})
    start_date = training_config.get('start_date', '2020-01-01')
    end_date = training_config.get('end_date', '2024-12-31')
    
    with Timer("总预测时间"):
        predictor.run(start_date, end_date)
    
    logger.info("滚动预测流程结束")


if __name__ == '__main__':
    main()

