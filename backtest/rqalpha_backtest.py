"""
使用 RQAlpha 框架进行回测
基于 Qlib 模型预测结果选股并回测
"""
from sched import scheduler
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from numpy._core.multiarray import scalar
import pandas as pd
from datetime import datetime
from rqalpha.api import *
from rqalpha import run_func
from utils.logger import setup_logger, default_logger
from utils.tools import load_yaml


class ModelPredictionStrategy:
    """基于模型预测的选股策略"""
    
    def __init__(self, prediction_dir='data/predictions', top_n=10):
        self.prediction_dir = prediction_dir
        self.top_n = top_n
        self.predictions_cache = {}
        
    def load_predictions(self, date_str):
        """加载预测文件"""
        if date_str in self.predictions_cache:
            return self.predictions_cache[date_str]
        
        pred_file = f"{self.prediction_dir}/pred_{date_str}.csv"
        
        if not os.path.exists(pred_file):
            default_logger.warning(f"预测文件不存在: {pred_file}")
            return None
        
        try:
            predictions = pd.read_csv(pred_file)
            # 标准化股票代码格式为 RQAlpha 格式（6位代码.交易所）
            predictions['rq_symbol'] = predictions['instrument'].apply(self.to_rqalpha_symbol)
            self.predictions_cache[date_str] = predictions
            return predictions
        except Exception as e:
            default_logger.error(f"加载预测文件失败: {e}")
            return None
    
    def to_rqalpha_symbol(self, instrument):
        """
        转换股票代码为 RQAlpha 格式
        000001 -> 000001.XSHE (深交所)
        600000 -> 600000.XSHG (上交所)
        """
        code = str(instrument).strip()
        
        # 移除可能存在的后缀
        if '.' in code:
            code = code.split('.')[0]
        
        # 确保是6位数字
        code = code[:6].zfill(6)
        
        # 根据代码判断交易所
        if code.startswith('6') or code.startswith('9'):
            return f"{code}.XSHG"  # 上交所
        elif code.startswith('0') or code.startswith('3'):
            return f"{code}.XSHE"  # 深交所
        else:
            return code


def init(context):
    """策略初始化"""
    # 加载配置
    config = load_yaml('config/backtest.yaml')
    
    context.strategy = ModelPredictionStrategy(
        prediction_dir='data/predictions',
        top_n=config.get('strategy', {}).get('top_n', 10)
    )
    
    # 设置调仓每日
    scheduler.run_daily(rebalance)
    # 设置调仓频率（每周一）
    #scheduler.run_weekly(rebalance, weekday=1)
    
    default_logger.info("=" * 80)
    default_logger.info("RQAlpha 回测初始化完成")
    default_logger.info(f"选股数量: {context.strategy.top_n}")
    default_logger.info("=" * 80)


def rebalance(context, bar_dict):
    """每周调仓"""
    # 获取当前日期
    current_date = context.now.strftime('%Y-%m-%d')
    default_logger.info(f"\n{'='*80}")
    default_logger.info(f"调仓日期: {current_date} ({context.now.strftime('%A')})")
    
    # 加载预测
    predictions = context.strategy.load_predictions(current_date)
    
    if predictions is None or len(predictions) == 0:
        default_logger.warning(f"日期 {current_date} 无预测数据，跳过调仓")
        return
    
    # 按分数排序，选择 top_n
    predictions = predictions.sort_values('score', ascending=False)
    selected = predictions.head(context.strategy.top_n)
    
    default_logger.info(f"selected ============= {selected}")
    default_logger.info(f"预测股票数: {len(predictions)}")
    default_logger.info(f"选中股票数: {len(selected)}")
    default_logger.info(f"\nTop {context.strategy.top_n} 股票:")
    for idx, row in selected.iterrows():
        default_logger.info(f"  {row['rq_symbol']}: 得分 {row['score']:.6f}")
    
    # 获取选中股票的 RQAlpha 代码
    target_stocks = selected['rq_symbol'].tolist()
    
    # 计算目标权重（等权重）
    target_weight = 1.0 / len(target_stocks) if len(target_stocks) > 0 else 0
    
    # 获取当前持仓
    current_positions = context.portfolio.positions # 获取持仓列表
    current_stocks = set(current_positions.keys())  # 获取持仓股票代码
    target_stocks_set = set(target_stocks)  # 获取目标股票代码
    
    # 记录调仓信息
    to_sell = current_stocks - target_stocks_set  
    to_buy = target_stocks_set - current_stocks # 获取需要买入的股票代码
    to_keep = current_stocks & target_stocks_set # 获取需要保持的股票代码
    
    default_logger.info(f"\n持仓调整:")
    default_logger.info(f"  保持持仓: {len(to_keep)} 只")
    default_logger.info(f"  需要卖出: {len(to_sell)} 只")
    default_logger.info(f"  需要买入: {len(to_buy)} 只")
    
    # 先卖出不在目标列表中的股票
    if to_sell:
        default_logger.info(f"\n卖出股票:")
        for stock in to_sell:
            if stock in current_positions and current_positions[stock].quantity > 0:
                default_logger.info(f"  卖出 {stock}: {current_positions[stock].quantity} 股")
                order_target_percent(stock, 0)
    
    # 买入目标股票（等权重）
    if target_stocks:
        default_logger.info(f"\n目标持仓（等权重 {target_weight:.2%}）:")
        for stock in target_stocks:
            try:
                current_percent = 0
                if stock in current_positions:
                    current_percent = current_positions[stock].market_value / context.portfolio.total_value
                
                default_logger.info(f"  {stock}: 目标 {target_weight:.2%}, 当前 {current_percent:.2%}")
                order_target_percent(stock, target_weight)
            except Exception as e:
                default_logger.warning(f"  {stock} 调仓失败: {e}")
    
    default_logger.info("=" * 80)


def handle_bar(context, bar_dict):
    """每日更新（可选）"""
    pass


def before_trading(context):
    """开盘前（可选）"""
    pass


def after_trading(context):
    """收盘后记录持仓信息"""
    positions = context.portfolio.positions
    
    if len(positions) > 0 and context.now.weekday() == 0:  # 周一
        default_logger.info(f"\n当日收盘持仓 ({context.now.strftime('%Y-%m-%d')}):")
        default_logger.info(f"{'股票代码':<15} {'数量':>10} {'成本价':>10} {'现价':>10} {'市值':>12} {'盈亏':>10}")
        default_logger.info("-" * 80)
        
        total_value = 0
        total_pnl = 0
        for stock, position in positions.items():
            pnl = (position.last_price - position.avg_price) * position.quantity
            total_value += position.market_value
            total_pnl += pnl
            
            default_logger.info(
                f"{stock:<15} {position.quantity:>10.0f} {position.avg_price:>10.2f} "
                f"{position.last_price:>10.2f} {position.market_value:>12.2f} {pnl:>10.2f}"
            )
        
        default_logger.info("-" * 80)
        default_logger.info(f"{'合计':<15} {'':<10} {'':<10} {'':<10} {total_value:>12.2f} {total_pnl:>10.2f}")
        default_logger.info(f"现金: {context.portfolio.cash:,.2f}")
        default_logger.info(f"总资产: {context.portfolio.total_value:,.2f}")
        default_logger.info("=" * 80)


def run_rqalpha_backtest():
    """运行 RQAlpha 回测"""
    # 设置日志
    logger = setup_logger(
        'rqalpha_backtest',
        f'logs/rqalpha_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    
    # 加载配置
    backtest_config = load_yaml('config/backtest.yaml')
    data_config = load_yaml('config/data.yaml')
    
    # RQAlpha 配置
    config = {
        "base": {
            "start_date": backtest_config['backtest']['start_date'],
            "end_date": backtest_config['backtest']['end_date'],
            "benchmark": "000300.XSHG",  # 沪深300作为基准
            "accounts": {
                "stock": backtest_config['backtest']['capital']
            },
            "frequency": "1d",
            "matching_type": "current_bar",  # 当日收盘价成交
        },
        "extra": {
            "log_level": "info",
        },
        "mod": {
            "sys_analyser": {
                "enabled": True,
                "plot": False,  # 不自动绘图
                "report_save_path": "data/backtest_results/rqalpha_report.pkl",
            },
            "sys_progress": {
                "enabled": True,
                "show": True,
            },
            "sys_risk": {
                "enabled": True,
            },
            "sys_simulation": {
                "enabled": True,
                "commission_multiplier": backtest_config['costs']['commission'] / 0.0003,  # 相对于默认佣金的倍数
                "slippage": backtest_config['costs']['slippage'],
            },
        }
    }
    
    default_logger.info("开始 RQAlpha 回测")
    default_logger.info(f"回测区间: {config['base']['start_date']} ~ {config['base']['end_date']}")
    default_logger.info(f"初始资金: {config['base']['accounts']['stock']:,.0f}")
    default_logger.info(f"基准指数: {config['base']['benchmark']}")
    
    # 运行回测
    try:
        results = run_func(
            init=init,
            handle_bar=handle_bar,
            before_trading=before_trading,
            after_trading=after_trading,
            config=config
        )
        
        # 提取回测报告
        if 'sys_analyser' in results:
            report = results['sys_analyser']
            
            # 打印回测结果摘要
            print_backtest_summary(report)
            
            # 保存详细结果
            save_detailed_results(report)
            
            logger.info("RQAlpha 回测完成")
            return report
        else:
            logger.error("未能获取回测报告")
            return None
            
    except Exception as e:
        logger.error(f"回测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_backtest_summary(report):
    """打印回测结果摘要"""
    summary = report['summary']
    
    print("\n" + "=" * 80)
    print("RQAlpha 回测结果摘要")
    print("=" * 80)
    
    print(f"\n总收益率: {summary['total_returns']:.2%}")
    print(f"年化收益率: {summary['annualized_returns']:.2%}")
    print(f"基准收益率: {summary['benchmark_total_returns']:.2%}")
    print(f"Alpha: {summary['alpha']:.4f}")
    print(f"Beta: {summary['beta']:.4f}")
    print(f"夏普比率: {summary['sharpe']:.4f}")
    print(f"最大回撤: {summary['max_drawdown']:.2%}")
    print(f"波动率: {summary['volatility']:.2%}")
    
    print(f"\n交易统计:")
    print(f"总交易次数: {summary.get('total_trades', 0)}")
    print(f"胜率: {summary.get('win_rate', 0):.2%}")
    
    print("=" * 80)


def save_detailed_results(report):
    """保存详细的回测结果"""
    output_dir = 'data/backtest_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存权益曲线
    if 'portfolio' in report:
        portfolio_df = report['portfolio']
        portfolio_df.to_csv(f'{output_dir}/rqalpha_portfolio.csv')
        default_logger.info(f"权益曲线已保存: {output_dir}/rqalpha_portfolio.csv")
    
    # 保存交易记录
    if 'trades' in report:
        trades_df = report['trades']
        trades_df.to_csv(f'{output_dir}/rqalpha_trades.csv')
        default_logger.info(f"交易记录已保存: {output_dir}/rqalpha_trades.csv")
        
        # 打印交易明细
        print("\n" + "=" * 80)
        print("交易明细（前20条）")
        print("=" * 80)
        print(trades_df.head(20).to_string())
    
    # 保存持仓记录
    if 'stock_positions' in report:
        positions_df = report['stock_positions']
        positions_df.to_csv(f'{output_dir}/rqalpha_positions.csv')
        default_logger.info(f"持仓记录已保存: {output_dir}/rqalpha_positions.csv")
    
    # 保存摘要
    if 'summary' in report:
        summary_df = pd.DataFrame([report['summary']])
        summary_df.to_csv(f'{output_dir}/rqalpha_summary.csv', index=False)
        default_logger.info(f"回测摘要已保存: {output_dir}/rqalpha_summary.csv")


if __name__ == '__main__':
    run_rqalpha_backtest()

