"""vn.py真实回测引擎集成"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict

import pandas as pd
import numpy as np
import polars as pl

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import qlib
from qlib.data import D
from vnpy.alpha.lab import AlphaLab
from vnpy.alpha.strategy.backtesting import BacktestingEngine
from vnpy.alpha.strategy.template import AlphaStrategy
from vnpy.trader.object import BarData, TradeData, OrderData
from vnpy.trader.constant import Direction, Offset, Interval, Status
from vnpy.trader.utility import extract_vt_symbol

from utils.logger import setup_logger, default_logger
from utils.tools import load_yaml, ensure_dir


class QlibDataAdapter:
    """Qlib数据适配器，将Qlib数据转换为vn.py需要的格式"""
    
    def __init__(self, data_config, lab_path="data/vnpy_lab"):
        self.data_config = data_config
        self.qlib_config = data_config.get('qlib', {})
        self.provider_uri = self.qlib_config.get('provider_uri', 'D:/qlib_data/qlib_data')
        self.region = self.qlib_config.get('region', 'cn')
        self.lab_path = lab_path
        self.qlib_initialized = False
        
        # 初始化Qlib
        self.init_qlib()
        
        # 初始化AlphaLab
        ensure_dir(lab_path)
        self.lab = AlphaLab(lab_path)
    
    def init_qlib(self):
        """初始化Qlib"""
        if not self.qlib_initialized:
            try:
                qlib.init(provider_uri=self.provider_uri, region=self.region)
                self.qlib_initialized = True
                default_logger.info(f"Qlib初始化成功: {self.provider_uri}")
            except Exception as e:
                default_logger.error(f"Qlib初始化失败: {e}")
                raise
    
    def load_instruments_from_file(self, file_path):
        """从文件中读取股票代码列表"""
        if not os.path.exists(file_path):
            default_logger.warning(f"股票代码文件不存在: {file_path}")
            return []
        
        instruments = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        code = line.split('\t')[0].strip()
                        if code.startswith('SH') or code.startswith('SZ'):
                            code = code[2:]
                        if code:
                            instruments.append(code[:6])
            return instruments
        except Exception as e:
            default_logger.error(f"读取股票代码文件失败: {e}")
            return []
    
    def normalize_instrument(self, instrument):
        """标准化股票代码格式为vn.py格式"""
        instrument = str(instrument).strip()
        
        # 如果已经包含交易所后缀，先转换为标准格式
        if '.' in instrument:
            code, exchange = instrument.split('.')
            # 转换为vn.py标准交易所代码
            if exchange.upper() in ['SH', 'SSE', 'SHSE']:
                return f"{code}.SSE"
            elif exchange.upper() in ['SZ', 'SZSE']:
                return f"{code}.SZSE"
            else:
                return instrument
        
        if instrument.startswith('SH') or instrument.startswith('SZ'):
            instrument = instrument[2:]
        
        if len(instrument) > 6:
            instrument = instrument[:6]
        
        # 根据代码判断交易所，使用vn.py标准代码
        if instrument.startswith('6') or instrument.startswith('9'):
            return f"{instrument}.SSE"  # 上交所
        elif instrument.startswith('0') or instrument.startswith('3'):
            return f"{instrument}.SZSE"  # 深交所
        else:
            return instrument
    
    def load_bar_data_from_qlib(
        self, 
        instruments: List[str], 
        start: datetime, 
        end: datetime
    ) -> Dict[str, List[BarData]]:
        """从Qlib加载K线数据并转换为BarData"""
        default_logger.info(f"开始从Qlib加载数据: {len(instruments)}只股票, {start} ~ {end}")
        
        # 标准化股票代码
        vt_symbols = [self.normalize_instrument(inst) for inst in instruments]
        
        # 从Qlib获取数据
        try:
            data = D.features(
                instruments=instruments,
                fields=['$open', '$high', '$low', '$close', '$volume', '$factor'],
                start_time=start.strftime('%Y-%m-%d'),
                end_time=end.strftime('%Y-%m-%d'),
                freq='day'
            )
        except Exception as e:
            default_logger.error(f"从Qlib获取数据失败: {e}")
            return {}
        
        if data.empty:
            default_logger.warning("Qlib返回数据为空")
            return {}
        
        # 转换为BarData
        bars_dict = {}
        
        for vt_symbol, instrument in zip(vt_symbols, instruments):
            try:
                # 提取该股票的数据
                if isinstance(data.index, pd.MultiIndex):
                    stock_data = data.loc[data.index.get_level_values('instrument') == instrument]
                else:
                    stock_data = data[data.index == instrument]
                
                if stock_data.empty:
                    continue
                
                bars = []
                symbol, exchange = extract_vt_symbol(vt_symbol)
                
                for idx, row in stock_data.iterrows():
                    # 处理MultiIndex
                    if isinstance(idx, tuple):
                        dt = idx[1] if len(idx) > 1 else idx[0]
                    else:
                        dt = idx
                    
                    # 转换为datetime
                    if not isinstance(dt, pd.Timestamp):
                        dt = pd.to_datetime(dt)
                    
                    # 处理复权因子
                    factor = row.get('factor', 1.0) if 'factor' in row else 1.0
                    if pd.isna(factor) or factor == 0:
                        factor = 1.0
                    
                    # 获取价格数据
                    open_price = float(row.get('open', row.get('$open', 0)) * factor)
                    high_price = float(row.get('high', row.get('$high', 0)) * factor)
                    low_price = float(row.get('low', row.get('$low', 0)) * factor)
                    close_price = float(row.get('close', row.get('$close', 0)) * factor)
                    volume = float(row.get('volume', row.get('$volume', 0)))
                    
                    if close_price <= 0:
                        continue
                    
                    bar = BarData(
                        symbol=symbol,
                        exchange=exchange,
                        datetime=dt,
                        interval=Interval.DAILY,
                        open_price=open_price,
                        high_price=high_price,
                        low_price=low_price,
                        close_price=close_price,
                        volume=volume,
                        turnover=volume * close_price,
                        open_interest=0.0,
                        gateway_name="QLIB"
                    )
                    bars.append(bar)
                
                if bars:
                    bars_dict[vt_symbol] = bars
                    # 保存到AlphaLab
                    self.lab.save_bar_data(bars)
                    default_logger.info(f"加载 {vt_symbol}: {len(bars)} 条K线数据")
            
            except Exception as e:
                default_logger.warning(f"处理 {vt_symbol} 数据失败: {e}")
                continue
        
        default_logger.info(f"成功加载 {len(bars_dict)} 只股票的K线数据")
        return bars_dict
    
    def create_contract_settings(self, vt_symbols: List[str]) -> Dict:
        """创建合约配置"""
        settings = {}
        for vt_symbol in vt_symbols:
            settings[vt_symbol] = {
                "size": 100,  # 每手100股
                "pricetick": 0.01,  # 最小价格变动
                "long_rate": 0.0003,  # 做多手续费率
                "short_rate": 0.0003  # 做空手续费率
            }
        return settings
    
    def save_contract_settings(self, settings: Dict):
        """保存合约配置"""
        import json
        contract_path = Path(self.lab_path) / "contract.json"
        with open(contract_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        default_logger.info(f"合约配置已保存: {contract_path}")


class WeeklyRotationAlphaStrategy(AlphaStrategy):
    """周频轮动Alpha策略"""
    
    def __init__(self, strategy_engine, strategy_name, vt_symbols, setting):
        super().__init__(strategy_engine, strategy_name, vt_symbols, setting)
        
        # 策略参数
        self.top_n = setting.get('top_n', 20)
        self.max_single_weight = setting.get('max_single_weight', 0.10)
        self.max_turnover = setting.get('max_turnover', 0.40)
        self.prediction_dir = setting.get('prediction_dir', 'data/predictions')
        self.freq = setting.get('freq', 'day')
        # 状态
        self.current_date = None
        self.last_rebalance_date = None
        self.predictions_cache = {}
    
    def on_init(self) -> None:
        """策略初始化"""
        default_logger.info("周频轮动策略初始化")
    
    def on_bars(self, bars: Dict[str, BarData]) -> None:
        """K线更新回调"""
        if not bars:
            return
        
        # 获取当前日期（使用第一个bar的日期）
        first_bar = list(bars.values())[0]
        current_date = first_bar.datetime.date()
        
        if self.freq == 'day':
            # if self.last_rebalance_date != current_date:
                self.rebalance(bars, current_date)
                # self.last_rebalance_date = current_date
        elif self.freq == 'week':
            # 判断是否周一（需要调仓）
            if current_date.weekday() == 0:  # 周一
                if self.last_rebalance_date != current_date:
                    self.rebalance(bars, current_date)
                    self.last_rebalance_date = current_date
            else:
                self.current_date = current_date
    
    def load_predictions(self, date: datetime.date) -> pd.DataFrame:
        """加载预测文件"""
        date_str = date.strftime('%Y-%m-%d')
        
        if date_str in self.predictions_cache:
            return self.predictions_cache[date_str]
        
        pred_file = f"{self.prediction_dir}/pred_{date_str}.csv"
        
        if not os.path.exists(pred_file):
            return None
        
        try:
            predictions = pd.read_csv(pred_file)
            self.predictions_cache[date_str] = predictions
            return predictions
        except Exception as e:
            default_logger.warning(f"加载预测文件失败: {e}")
            return None
    
    def rebalance(self, bars: Dict[str, BarData], date: datetime.date):
        """调仓"""
        default_logger.info(f"调仓日期: {date}")
        
        # 加载预测
        predictions = self.load_predictions(date)
        if predictions is None:
            default_logger.warning("无预测数据，跳过调仓")
            return

        
        # 选股
        predictions = predictions.sort_values('score', ascending=False)
        selected = predictions.head(self.top_n)
        
        if len(selected) == 0:
            default_logger.warning("无选中股票")
            return
        
        # 计算目标权重（等权重）
        n_stocks = len(selected)
        weight = min(1.0 / n_stocks, self.max_single_weight)
        
        # 获取当前持仓价值
        current_value = self.calculate_portfolio_value(bars)
        if current_value <= 0:
            current_value = self.strategy_engine.capital
        
        # 计算目标持仓
        target_positions = {}
        selected_vt_symbols = set()
        
        for _, row in selected.iterrows():
            instrument = row['instrument']
            vt_symbol = self.normalize_instrument(instrument)
            selected_vt_symbols.add(vt_symbol)
            
            if vt_symbol not in bars:
                default_logger.warning(f"股票 {vt_symbol} 无K线数据，跳过")
                continue
            
            bar = bars[vt_symbol]
            
            # 检查价格是否有效
            if bar.close_price is None or pd.isna(bar.close_price) or bar.close_price <= 0:
                default_logger.warning(f"股票 {vt_symbol} 收盘价无效: {bar.close_price}，跳过")
                continue
            
            target_value = current_value * weight
            target_volume = int(target_value / bar.close_price / 100) * 100  # 转换为手
            
            if target_volume > 0:
                target_positions[vt_symbol] = target_volume
        
        # 先卖出不在选中列表中的持仓
        for vt_symbol in list(self.pos_data.keys()):
            if vt_symbol not in selected_vt_symbols and self.pos_data[vt_symbol] > 0:
                if vt_symbol in bars:
                    bar = bars[vt_symbol]
                    # 检查价格是否有效
                    if bar.close_price is None or pd.isna(bar.close_price) or bar.close_price <= 0:
                        default_logger.warning(f"股票 {vt_symbol} 收盘价无效，无法卖出")
                        continue
                    self.sell(vt_symbol, bar.close_price, self.pos_data[vt_symbol])
        
        # 再调整持仓到目标
        for vt_symbol, target_volume in target_positions.items():
            current_pos = self.pos_data.get(vt_symbol, 0)
            
            if target_volume > current_pos:
                # 买入
                buy_volume = target_volume - current_pos
                if vt_symbol in bars:
                    bar = bars[vt_symbol]
                    # 检查价格是否有效
                    if bar.close_price is None or pd.isna(bar.close_price) or bar.close_price <= 0:
                        default_logger.warning(f"股票 {vt_symbol} 收盘价无效，无法买入")
                        continue
                    self.buy(vt_symbol, bar.close_price, buy_volume)
                    default_logger.info(f"买入: {vt_symbol}, 价格: {bar.close_price}, 数量: {buy_volume}")
            elif target_volume < current_pos:
                # 卖出
                sell_volume = current_pos - target_volume
                if vt_symbol in bars:
                    bar = bars[vt_symbol]
                    # 检查价格是否有效
                    if bar.close_price is None or pd.isna(bar.close_price) or bar.close_price <= 0:
                        default_logger.warning(f"股票 {vt_symbol} 收盘价无效，无法卖出")
                        continue
                    self.sell(vt_symbol, bar.close_price, sell_volume)
                    default_logger.info(f"卖出: {vt_symbol}, 价格: {bar.close_price}, 数量: {sell_volume}")
    
    def normalize_instrument(self, instrument):
        """标准化股票代码为vn.py格式"""
        instrument = str(instrument).strip()
        
        # 如果已经包含交易所后缀，转换为vn.py标准格式
        if '.' in instrument:
            code, exchange = instrument.split('.')
            if exchange.upper() in ['SH', 'SSE', 'SHSE']:
                return f"{code}.SSE"
            elif exchange.upper() in ['SZ', 'SZSE']:
                return f"{code}.SZSE"
            else:
                return instrument
        
        if instrument.startswith('SH') or instrument.startswith('SZ'):
            instrument = instrument[2:]
        if len(instrument) > 6:
            instrument = instrument[:6]
        
        # 根据代码判断交易所
        if instrument.startswith('6') or instrument.startswith('9'):
            return f"{instrument}.SSE"
        elif instrument.startswith('0') or instrument.startswith('3'):
            return f"{instrument}.SZSE"
        return instrument
    
    def calculate_portfolio_value(self, bars: Dict[str, BarData]) -> float:
        """计算组合价值"""
        value = 0
        for vt_symbol, pos in self.pos_data.items():
            if vt_symbol in bars:
                bar = bars[vt_symbol]
                # 检查价格是否有效
                if bar.close_price is not None and not pd.isna(bar.close_price) and bar.close_price > 0:
                    value += pos * bar.close_price
                else:
                    default_logger.warning(f"股票 {vt_symbol} 收盘价无效: {bar.close_price}，无法计算持仓价值")
        return value
    
    def on_trade(self, trade: TradeData) -> None:
        """成交回调"""
        default_logger.info(f"成交: {trade.vt_symbol}, 方向: {trade.direction}, "
                          f"价格: {trade.price}, 数量: {trade.volume}")


class VnpyBacktester:
    """vn.py真实回测引擎封装"""
    
    def __init__(self, config, data_config):
        self.config = config
        self.data_config = data_config
        self.backtest_config = config.get('backtest', {})
        self.strategy_config = config.get('strategy', {})
        
        # 创建数据适配器
        lab_path = config.get('output', {}).get('lab_path', 'data/vnpy_lab')
        self.adapter = QlibDataAdapter(data_config, lab_path)
        
        # 创建回测引擎
        self.engine = BacktestingEngine(self.adapter.lab)
        
        # 结果
        self.results = None
    
    def prepare_data(self, instruments: List[str], start: datetime, end: datetime):
        """准备回测数据"""
        default_logger.info("开始准备回测数据")
        
        # 从Qlib加载数据
        bars_dict = self.adapter.load_bar_data_from_qlib(instruments, start, end)
        
        if not bars_dict:
            raise ValueError("未能加载任何K线数据")
        
        # 创建合约配置
        vt_symbols = list(bars_dict.keys())
        settings = self.adapter.create_contract_settings(vt_symbols)
        self.adapter.save_contract_settings(settings)
        
        return vt_symbols
    
    def run(
        self, 
        instruments: List[str], 
        start: datetime, 
        end: datetime,
        capital: float = 10000000
    ):
        """运行回测"""
        default_logger.info("=" * 60)
        default_logger.info("开始vn.py回测")
        default_logger.info(f"股票数量: {len(instruments)}")
        default_logger.info(f"时间范围: {start} ~ {end}")
        default_logger.info(f"初始资金: {capital:,.0f}")
        default_logger.info("=" * 60)
        
        # 准备数据
        vt_symbols = self.prepare_data(instruments, start, end)
        
        # 设置回测参数
        self.engine.set_parameters(
            vt_symbols=vt_symbols,
            interval=Interval.DAILY,
            start=start,
            end=end,
            capital=capital,
            risk_free=0.0,
            annual_days=240
        )
        
        # 加载历史数据
        self.engine.load_data()
        
        # 创建信号DataFrame（空信号，策略内部处理）
        signal_df = pl.DataFrame({
            "datetime": [],
            "vt_symbol": [],
            "signal": []
        })
        
        # 添加策略
        strategy_setting = {
            'top_n': self.strategy_config.get('top_n', 20),
            'max_single_weight': self.strategy_config.get('max_single_weight', 0.10),
            'max_turnover': self.strategy_config.get('max_turnover', 0.40),
            'prediction_dir': 'data/predictions'
        }
        
        self.engine.add_strategy(
            WeeklyRotationAlphaStrategy,
            strategy_setting,
            signal_df
        )
        
        # 运行回测
        self.engine.run_backtesting()
        
        # 计算结果
        self.engine.calculate_result()
        stats = self.engine.calculate_statistics()
        
        # 保存结果
        self.save_results()
        
        default_logger.info("vn.py回测完成")
        return stats
    
    def save_results(self):
        """保存回测结果"""
        output_dir = self.config.get('output', {}).get('report_path', 'data/backtest_results')
        ensure_dir(output_dir)
        
        # 保存每日结果
        if hasattr(self.engine, 'daily_df') and self.engine.daily_df is not None:
            df = self.engine.daily_df.to_pandas()
            df.to_csv(f"{output_dir}/equity_curve.csv", index=False)
        
        # 保存交易记录
        if self.engine.trades:
            trades_data = []
            for trade in self.engine.trades.values():
                trades_data.append({
                    'datetime': trade.datetime,
                    'vt_symbol': trade.vt_symbol,
                    'direction': trade.direction.value,
                    'price': trade.price,
                    'volume': trade.volume
                })
            pd.DataFrame(trades_data).to_csv(f"{output_dir}/trades.csv", index=False)
        
        default_logger.info(f"回测结果已保存到: {output_dir}")


def main():
    """主函数"""
    logger = setup_logger(
        'vnpy_backtest',
        f'logs/vnpy_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    
    # 加载配置
    config = load_yaml('config/backtest.yaml')
    data_config = load_yaml('config/data.yaml')
    
    # 获取股票列表
    provider_uri = data_config['qlib']['provider_uri']
    csi300_file = os.path.join(provider_uri, 'instruments', 'csi300.txt')
    
    adapter = QlibDataAdapter(data_config)
    instruments = adapter.load_instruments_from_file(csi300_file)
    if not instruments:
        instruments = ['csi300']  # 使用默认
    
    # 创建回测器
    backtester = VnpyBacktester(config, data_config)
    
    # 运行回测
    backtest_config = config.get('backtest', {})
    start_date = datetime.strptime(backtest_config.get('start_date', '2020-01-01'), '%Y-%m-%d')
    end_date = datetime.strptime(backtest_config.get('end_date', '2024-12-31'), '%Y-%m-%d')
    capital = backtest_config.get('capital', 10000000)
    
    stats = backtester.run(instruments, start_date, end_date, capital)
    
    logger.info("回测完成")
    return stats


if __name__ == '__main__':
    main()

