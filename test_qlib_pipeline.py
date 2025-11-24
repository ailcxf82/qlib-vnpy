"""测试 qlib feature pipeline"""
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from feature.qlib_feature_pipeline import QlibFeaturePipeline
from utils.logger import setup_logger, default_logger
from utils.tools import load_yaml


def test_qlib_pipeline():
    """测试 qlib feature pipeline"""
    # 设置日志
    setup_logger('test_qlib_pipeline', 'logs/test_qlib_pipeline.log')
    
    # 加载配置
    data_config = load_yaml('config/data.yaml')
    pipeline_config = load_yaml('config/pipeline.yaml')
    
    # 创建 pipeline
    pipeline = QlibFeaturePipeline(data_config, pipeline_config)
    
    # 测试时间范围
    start_time = '2024-01-01'
    end_time = '2024-08-31'
    
    default_logger.info("=" * 80)
    default_logger.info("测试 1: 获取 DataFrame 格式的特征和标签")
    default_logger.info("=" * 80)
    
    try:
        features, labels = pipeline.run(
            start_time=start_time,
            end_time=end_time,
            instruments='csi300_file',
            return_dataset=False
        )
        
        default_logger.info(f"✓ 成功获取特征和标签")
        default_logger.info(f"  特征形状: {features.shape}")
        default_logger.info(f"  标签形状: {labels.shape}")
        default_logger.info(f"  特征列数: {len(features.columns)}")
        default_logger.info(f"  特征列示例: {list(features.columns[:5])}")
        
    except Exception as e:
        default_logger.error(f"✗ 测试失败: {e}", exc_info=True)
    
    default_logger.info("")
    default_logger.info("=" * 80)
    default_logger.info("测试 2: 获取 qlib Dataset 对象")
    default_logger.info("=" * 80)
    
    try:
        dataset = pipeline.run(
            start_time=start_time,
            end_time=end_time,
            instruments='csi300_file',
            return_dataset=True
        )
        
        default_logger.info(f"✓ 成功创建 qlib Dataset")
        default_logger.info(f"  Dataset 类型: {type(dataset)}")
        default_logger.info(f"  Handler 类型: {type(dataset.handler)}")
        
        # 测试获取数据
        features = dataset.handler.fetch(col_set="feature", data_key="infer")
        labels = dataset.handler.fetch(col_set="label", data_key="infer")
        
        default_logger.info(f"  从 Dataset 获取的特征形状: {features.shape}")
        default_logger.info(f"  从 Dataset 获取的标签形状: {labels.shape}")
        
    except Exception as e:
        default_logger.error(f"✗ 测试失败: {e}", exc_info=True)
    
    default_logger.info("")
    default_logger.info("=" * 80)
    default_logger.info("测试完成")
    default_logger.info("=" * 80)


if __name__ == '__main__':
    test_qlib_pipeline()


