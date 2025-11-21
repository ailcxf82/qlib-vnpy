"""特征处理流程"""
import pandas as pd
from factors.factor_engine import FactorEngine
from feature.label import LabelGenerator
from feature.dataset_builder import DatasetBuilder
from utils.logger import default_logger
from utils.timer import timing_decorator
from utils.tools import ensure_dir


class FeaturePipeline:
    """特征处理流程"""
    
    def __init__(self, data_config, pipeline_config):
        self.data_config = data_config
        self.pipeline_config = pipeline_config
        
        self.factor_engine = FactorEngine(data_config)
        self.label_generator = LabelGenerator(pipeline_config)
        self.dataset_builder = DatasetBuilder(pipeline_config)
    
    @timing_decorator
    def run(self, start_time, end_time, instruments='csi300_file'):
        """
        运行完整的特征生成流程
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            instruments: 股票池（'csi300_file'从文件读取，'csi300'使用Qlib内置，'all'全市场）
        
        Returns:
            features, labels
        """
        default_logger.info(f"开始特征生成流程: {start_time} to {end_time}")
        
        # 1. 计算因子
        factors = self.factor_engine.calculate_factors(
            start_time=start_time,
            end_time=end_time,
            instruments=instruments
        )
        
        if factors is None or len(factors) == 0:
            default_logger.error("因子计算失败")
            return None, None
        
        # 2. 获取价格数据用于生成标签
        # 扩展字段以支持更丰富的标签生成
        price_data = self.factor_engine.fetch_qlib_data(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            fields=['$open', '$high', '$low', '$close', '$volume', '$factor']
        )
        
        # 3. 生成标签
        label_config = self.pipeline_config.get('label', {})
        forward_days = label_config.get('forward_days', 5)
        label_type = label_config.get('type', 'return')  # 支持多种标签类型
        
        default_logger.info(f"标签类型: {label_type}, 向前天数: {forward_days}")
        
        # 使用统一接口生成标签
        labels = self.label_generator.generate_label(
            price_data,
            label_type=label_type,
            forward_days=forward_days
        )
        
        # 4. 构建数据集
        features, labels = self.dataset_builder.build_dataset(
            factors,
            labels,
            preprocess=True
        )
        
        default_logger.info("特征生成流程完成")
        
        return features, labels
    
    def save(self, features, labels, output_dir):
        """保存特征和标签"""
        ensure_dir(output_dir)
        
        feature_path = f"{output_dir}/features.pkl"
        label_path = f"{output_dir}/labels.pkl"
        
        features.to_pickle(feature_path)
        labels.to_pickle(label_path)
        
        default_logger.info(f"特征和标签已保存到: {output_dir}")
    
    def load(self, input_dir):
        """加载特征和标签"""
        feature_path = f"{input_dir}/features.pkl"
        label_path = f"{input_dir}/labels.pkl"
        
        features = pd.read_pickle(feature_path)
        labels = pd.read_pickle(label_path)
        
        default_logger.info(f"特征和标签已从 {input_dir} 加载")
        
        return features, labels

