"""一键运行训练+预测"""
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)  # 切换工作目录到项目根目录

from datetime import datetime
from pipeline.run_train import RollingTrainer
from pipeline.run_predict import RollingPredictor
from utils.logger import setup_logger
from utils.timer import Timer
from utils.tools import load_yaml


def main():
    """主函数"""
    # 设置日志
    logger = setup_logger(
        'run_all',
        f'logs/run_all_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    
    logger.info("=" * 80)
    logger.info("开始完整的训练+预测流程")
    logger.info("=" * 80)
    
    # 加载配置
    data_config = load_yaml('config/data.yaml')
    pipeline_config = load_yaml('config/pipeline.yaml')
    model_config = load_yaml('config/model.yaml')
    
    # 获取时间范围
    training_config = pipeline_config.get('training', {})
    train_start = training_config.get('start_date', '2017-01-01')
    train_end = training_config.get('end_date', '2024-12-31')
    
    # 1. 滚动训练
    logger.info("\n" + "=" * 80)
    logger.info("阶段 1: 滚动训练")
    logger.info("=" * 80)
    
    trainer = RollingTrainer(data_config, pipeline_config, model_config)
    
    with Timer("滚动训练总时间"):
        trained_models = trainer.run(train_start, train_end)
    
    # 2. 滚动预测
    logger.info("\n" + "=" * 80)
    logger.info("阶段 2: 滚动预测")
    logger.info("=" * 80)
    
    # 从训练结束后开始预测
    predict_start = train_start  # 或者根据训练窗口自动计算
    predict_end = train_end
    
    predictor = RollingPredictor(data_config, pipeline_config, model_config)
    
    with Timer("滚动预测总时间"):
        predictions = predictor.run(predict_start, predict_end)
    
    # 3. 回测
    logger.info("\n" + "=" * 80)
    logger.info("阶段 3: 回测跳过")
    logger.info("=" * 80)
    
    # try:
    #     from backtest.run_vnpy_backtest import SimpleBacktester
    #     backtest_config = load_yaml('config/backtest.yaml')
        
    #     backtester = SimpleBacktester(backtest_config, data_config)
        
    #     backtest_start = backtest_config.get('backtest', {}).get('start_date', '2020-01-01')
    #     backtest_end = backtest_config.get('backtest', {}).get('end_date', '2024-12-31')
        
    #     with Timer("回测总时间"):
    #         backtester.run(backtest_start, backtest_end)
        
    #     logger.info("回测完成")
    # except Exception as e:
    #     logger.warning(f"回测失败: {e}")
    #     logger.info("跳过回测步骤")
    
    # 4. 总结
    logger.info("\n" + "=" * 80)
    logger.info("流程总结")
    logger.info("=" * 80)
    logger.info(f"成功训练模型数: {len(trained_models)}")
    logger.info(f"成功预测日期数: {len(predictions)}")
    logger.info("所有流程完成！")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

