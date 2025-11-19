"""使用已有模型进行预测并回测"""
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from datetime import datetime
from pipeline.run_predict import RollingPredictor
from backtest.run_vnpy_backtest import SimpleBacktester
from utils.logger import setup_logger, default_logger
from utils.timer import Timer
from utils.tools import load_yaml
import glob


def check_existing_predictions(pred_dir='data/predictions'):
    """检查已有预测文件"""
    pred_files = glob.glob(f"{pred_dir}/pred_*.csv")
    dates = []
    for f in pred_files:
        # 从文件名提取日期：pred_2020-01-06.csv
        date_str = os.path.basename(f).replace('pred_', '').replace('.csv', '')
        dates.append(date_str)
    dates.sort()
    return dates


def check_existing_models(model_dir='data/models'):
    """检查已有模型文件"""
    model_files = glob.glob(f"{model_dir}/model_*.txt")
    dates = []
    for f in model_files:
        # 从文件名提取日期：model_20200106.txt
        date_str = os.path.basename(f).replace('model_', '').replace('.txt', '')
        # 转换为标准格式：20200106 -> 2020-01-06
        if len(date_str) == 8:
            date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        dates.append(date_str)
    dates.sort()
    return dates


def main():
    """主函数"""
    # 设置日志
    logger = setup_logger(
        'predict_backtest',
        f'logs/predict_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    
    logger.info("=" * 80)
    logger.info("使用已有模型进行预测并回测")
    logger.info("=" * 80)
    
    # 加载配置
    data_config = load_yaml('config/data.yaml')
    pipeline_config = load_yaml('config/pipeline.yaml')
    model_config = load_yaml('config/model.yaml')
    backtest_config = load_yaml('config/backtest.yaml')
    
    # 检查已有模型和预测
    existing_models = check_existing_models()
    existing_predictions = check_existing_predictions()
    
    logger.info(f"已有模型数量: {len(existing_models)}")
    logger.info(f"已有预测数量: {len(existing_predictions)}")
    
    if len(existing_models) > 0:
        logger.info(f"模型日期范围: {existing_models[0]} ~ {existing_models[-1]}")
    if len(existing_predictions) > 0:
        logger.info(f"预测日期范围: {existing_predictions[0]} ~ {existing_predictions[-1]}")
    
    # 获取回测时间范围
    backtest_start = backtest_config.get('backtest', {}).get('start_date', '2020-01-01')
    backtest_end = backtest_config.get('backtest', {}).get('end_date', '2024-12-31')
    
    # 确定需要预测的日期（周一）
    # from utils.tools import get_trade_dates 
    # predict_dates = get_trade_dates(backtest_start, backtest_end, freq='W-MON')
    
    # 按交易日
    from utils.tools import get_real_trade_dates
    predict_dates = get_real_trade_dates(backtest_start, backtest_end)

    logger.info(f"回测时间范围: {backtest_start} ~ {backtest_end}")
    logger.info(f"需要预测的日期数: {len(predict_dates)}")
    
    # 检查哪些日期需要重新预测
    missing_predictions = []
    for date in predict_dates:
        pred_file = f"data/predictions/pred_{date}.csv"
        if not os.path.exists(pred_file):
            # 检查是否有对应的模型
            model_date = date.replace('-', '')
            model_file = f"data/models/model_{model_date}.txt"
            if os.path.exists(model_file):
                missing_predictions.append(date)
    
    # 如果需要，生成缺失的预测
    if missing_predictions:
        logger.info(f"发现 {len(missing_predictions)} 个日期需要生成预测")
        logger.info("开始生成预测...")
        
        predictor = RollingPredictor(data_config, pipeline_config, model_config)
        
        for date in missing_predictions:
            model_date = date.replace('-', '')
            model_path = f"data/models/model_{model_date}.txt"
            
            if os.path.exists(model_path):
                try:
                    logger.info(f"为日期 {date} 生成预测...")
                    predictor.predict_single_date(date, model_path, 'data')
                except Exception as e:
                    logger.error(f"预测日期 {date} 失败: {e}")
                    continue
    else:
        logger.info("所有预测文件已存在，跳过预测步骤")
    
    # 运行回测
    logger.info("\n" + "=" * 80)
    logger.info("开始回测")
    logger.info("=" * 80)
    
    backtester = SimpleBacktester(backtest_config, data_config)
    
    with Timer("回测总时间"):
        backtester.run(backtest_start, backtest_end)
    
    logger.info("=" * 80)
    logger.info("预测和回测流程完成！")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()



