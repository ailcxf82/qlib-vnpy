"""使用 qlib 标准方式处理特征和标签"""
import pandas as pd
import qlib
from qlib.data.dataset import DatasetH
from qlib.contrib.data.handler import Alpha158
from utils.logger import default_logger
from utils.timer import timing_decorator
from utils.tools import ensure_dir


class QlibFeaturePipeline:
    """使用 qlib 标准方式处理特征和标签的流程"""
    
    def __init__(self, data_config, pipeline_config):
        self.data_config = data_config
        self.pipeline_config = pipeline_config
        
        # 初始化 qlib
        self._init_qlib()
        
        # 标签配置
        self.label_config = pipeline_config.get('label', {})
    
    def _init_qlib(self):
        """初始化 Qlib"""
        try:
            provider_uri = self.data_config.get('qlib', {}).get('provider_uri')
            region = self.data_config.get('qlib', {}).get('region', 'cn')
            if provider_uri:
                qlib.init(provider_uri=provider_uri, region=region, silence=True)
                default_logger.info(f"Qlib 初始化成功: {provider_uri}")
        except Exception as e:
            default_logger.warning(f"Qlib 初始化失败（可能已初始化）: {e}")
    
    def _get_label_config(self):
        """
        获取标签配置
        
        Returns:
            tuple: (label_expr, label_name)
        """
        # 从配置中读取标签表达式
        label_expr = self.label_config.get('expr')
        label_name = self.label_config.get('name', 'LABEL0')
        
        # 如果没有配置表达式，使用默认的标签表达式
        if not label_expr:
            forward_days = self.label_config.get('forward_days', 5)
            # 默认标签：未来 N 天的收益率
            label_expr = f"Ref($close, -{forward_days}) / $close - 1"
            default_logger.info(f"使用默认标签表达式: {label_expr}")
        else:
            default_logger.info(f"使用配置的标签表达式: {label_expr}")
        
        return [label_expr], [label_name]
    
    def _load_instruments(self, instruments):
        """
        加载股票池
        
        Args:
            instruments: 股票池（'csi300_file'从文件读取，'csi300'使用Qlib内置，'all'全市场）
        
        Returns:
            股票池列表或字符串
        """
        if isinstance(instruments, str) and instruments == 'csi300_file':
            # 从文件读取股票代码列表
            from factors.factor_engine import FactorEngine
            factor_engine = FactorEngine(self.data_config)
            provider_uri = self.data_config.get('qlib', {}).get('provider_uri', '')
            csi300_file = f"{provider_uri}/instruments/csi300.txt"
            instruments = factor_engine.load_instruments_from_file(csi300_file)
            if not instruments:
                default_logger.warning("从文件读取股票代码为空，使用默认值 'csi300'")
                instruments = 'csi300'
        
        return instruments
    
    @timing_decorator
    def create_dataset(self, start_time, end_time, instruments='csi300_file'):
        """
        创建 qlib Dataset，包含特征和标签
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            instruments: 股票池
        
        Returns:
            DatasetH: qlib 数据集对象
        """
        default_logger.info(f"开始创建 qlib 数据集: {start_time} to {end_time}")
        
        # 加载股票池
        instruments = self._load_instruments(instruments)
        
        # 获取标签配置
        label_expr, label_name = self._get_label_config()
        
        # 创建 Alpha158 handler，并自定义标签
        handler = Alpha158(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            freq="day",
            label=(label_expr, label_name)  # 自定义标签表达式
        )
        
        default_logger.info(f"Alpha158 handler 创建完成")
        default_logger.info(f"特征数量: {len(handler.get_cols())}")
        
        # 创建 segments（将所有数据作为训练集）
        # DatasetH 需要 segments 参数，格式为 {'train': (start_date, end_date), ...}
        # 使用传入的时间范围作为训练集
        segments = {
            'train': (start_time, end_time)
        }
        
        # 创建数据集
        dataset = DatasetH(handler=handler, segments=segments)
        
        default_logger.info("qlib 数据集创建完成")
        return dataset
    
    @timing_decorator
    def get_features_and_labels(self, start_time, end_time, instruments='csi300_file', 
                                data_key='infer', col_set='feature'):
        """
        获取特征和标签数据（DataFrame 格式）
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            instruments: 股票池
            data_key: 数据键 ('raw', 'infer', 'learn')
            col_set: 列集合 ('feature', 'label', '__all')
        
        Returns:
            tuple: (features, labels) DataFrame
        """
        default_logger.info(f"开始获取特征和标签: {start_time} to {end_time}")
        
        # 创建数据集
        dataset = self.create_dataset(start_time, end_time, instruments)
        
        # 获取 handler
        handler = dataset.handler
        
        # 获取特征数据
        try:
            features = handler.fetch(col_set="feature", data_key=data_key)
            default_logger.info(f"特征数据形状: {features.shape}")
        except Exception as e:
            default_logger.warning(f"使用 col_set='feature' 失败: {e}，尝试获取所有数据")
            all_data = handler.fetch(data_key=data_key)
            if isinstance(all_data.columns, pd.MultiIndex):
                if 'feature' in all_data.columns.names:
                    features = all_data.xs('feature', axis=1, level=0)
                else:
                    features = all_data.droplevel(0, axis=1)
            else:
                # 排除标签列
                label_cols = [col for col in all_data.columns 
                            if 'LABEL' in str(col).upper() or 'label' in str(col).lower()]
                features = all_data.drop(columns=label_cols) if label_cols else all_data
        
        # 获取标签数据
        try:
            labels = handler.fetch(col_set="label", data_key=data_key)
            default_logger.info(f"标签数据形状: {labels.shape}")
        except Exception as e:
            default_logger.warning(f"使用 col_set='label' 失败: {e}，尝试从所有数据中提取")
            all_data = handler.fetch(data_key=data_key)
            if isinstance(all_data.columns, pd.MultiIndex):
                if 'label' in all_data.columns.names:
                    labels = all_data.xs('label', axis=1, level=0)
                else:
                    # 尝试查找标签列
                    label_cols = [col for col in all_data.columns.get_level_values(0) 
                                if 'LABEL' in str(col).upper() or 'label' in str(col).lower()]
                    if label_cols:
                        labels = all_data.xs(label_cols[0], axis=1, level=0)
                    else:
                        raise ValueError("无法找到标签列")
            else:
                # 查找标签列
                label_cols = [col for col in all_data.columns 
                            if 'LABEL' in str(col).upper() or 'label' in str(col).lower()]
                if label_cols:
                    labels = all_data[label_cols]
                else:
                    raise ValueError("无法找到标签列")
        
        # 确保标签是单列 DataFrame
        if isinstance(labels, pd.DataFrame):
            if len(labels.columns) == 1:
                labels = labels.iloc[:, 0]  # 转换为 Series
            else:
                # 使用第一列
                labels = labels.iloc[:, 0]
                default_logger.warning(f"标签有多列，使用第一列: {labels.name}")
        
        # 转换为 DataFrame 格式（与原有接口兼容）
        if isinstance(labels, pd.Series):
            labels = pd.DataFrame({'label': labels})
        
        default_logger.info(f"特征和标签获取完成")
        default_logger.info(f"特征形状: {features.shape}, 标签形状: {labels.shape}")
        
        return features, labels
    
    @timing_decorator
    def run(self, start_time, end_time, instruments='csi300_file', return_dataset=False):
        """
        运行完整的特征生成流程（兼容原有接口）
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            instruments: 股票池
            return_dataset: 是否返回 qlib Dataset 对象（默认 False，返回 DataFrame）
        
        Returns:
            如果 return_dataset=True: DatasetH 对象
            如果 return_dataset=False: (features, labels) DataFrame 元组
        """
        default_logger.info(f"开始特征生成流程: {start_time} to {end_time}")
        
        if return_dataset:
            # 返回 qlib Dataset 对象
            dataset = self.create_dataset(start_time, end_time, instruments)
            return dataset
        else:
            # 返回 DataFrame 格式（兼容原有接口）
            features, labels = self.get_features_and_labels(start_time, end_time, instruments)
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

