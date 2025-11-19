"""直接使用已有预测文件进行回测（跳过预测步骤）"""
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from datetime import datetime
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
        date_str = os.path.basename(f).replace('pred_', '').replace('.csv', '')
        dates.append(date_str)
    dates.sort()
    return dates


def main():
    """主函数"""
    # 设置日志
    logger = setup_logger(
        'backtest_existing',
        f'logs/backtest_existing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    
    logger.info("=" * 80)
    logger.info("使用已有预测文件进行回测")
    logger.info("=" * 80)
    
    # 加载配置
    data_config = load_yaml('config/data.yaml')
    backtest_config = load_yaml('config/backtest.yaml')
    
    # 检查已有预测文件
    existing_predictions = check_existing_predictions()
    
    logger.info(f"发现 {len(existing_predictions)} 个预测文件")
    if len(existing_predictions) > 0:
        logger.info(f"预测日期范围: {existing_predictions[0]} ~ {existing_predictions[-1]}")
    
    if len(existing_predictions) == 0:
        logger.error("未找到任何预测文件，请先运行预测流程")
        return
    
    # 获取回测时间范围
    backtest_start = backtest_config.get('backtest', {}).get('start_date', '2020-01-01')
    backtest_end = backtest_config.get('backtest', {}).get('end_date', '2024-12-31')
    
    # 检查预测文件是否覆盖回测时间范围
    if existing_predictions[0] > backtest_start:
        logger.warning(f"预测文件最早日期 {existing_predictions[0]} 晚于回测开始日期 {backtest_start}")
        logger.info(f"将回测开始日期调整为 {existing_predictions[0]}")
        backtest_start = existing_predictions[0]
    
    if existing_predictions[-1] < backtest_end:
        logger.warning(f"预测文件最晚日期 {existing_predictions[-1]} 早于回测结束日期 {backtest_end}")
        logger.info(f"将回测结束日期调整为 {existing_predictions[-1]}")
        backtest_end = existing_predictions[-1]
    
    logger.info(f"实际回测时间范围: {backtest_start} ~ {backtest_end}")
    
    # 运行回测
    logger.info("\n" + "=" * 80)
    logger.info("开始回测")
    logger.info("=" * 80)
    
    backtester = SimpleBacktester(backtest_config, data_config)
    
    with Timer("回测总时间"):
        backtester.run(backtest_start, backtest_end)
    
    logger.info("=" * 80)
    logger.info("回测完成！")
    logger.info("=" * 80)
    logger.info(f"回测结果保存在: data/backtest_results/")


if __name__ == '__main__':
    main()







