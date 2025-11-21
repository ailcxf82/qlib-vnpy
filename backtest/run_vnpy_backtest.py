"""vn.py回测主程序"""
import sys
import os
from pathlib import Path
import warnings

# 过滤 joblib resource_tracker 的警告（Windows系统常见问题，不影响功能）
# 使用多种方式确保过滤所有相关警告
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')
warnings.filterwarnings('ignore', message='.*resource_tracker.*')
warnings.filterwarnings('ignore', message='.*FileNotFoundError.*系统找不到指定的路径.*')

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)  # 切换工作目录到项目根目录

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import qlib
from qlib.data import D
from utils.logger import setup_logger, default_logger
from utils.tools import load_yaml, ensure_dir


class SimpleBacktester:
    """简单回测引擎（不依赖vn.py）"""
    
    def __init__(self, config, data_config=None):
        self.config = config
        self.data_config = data_config or {}
        self.backtest_config = config.get('backtest', {})
        self.strategy_config = config.get('strategy', {})
        self.costs_config = config.get('costs', {})
        self.risk_config = config.get('risk_control', {})
        
        # 参数
        self.capital = self.backtest_config.get('capital', 10000000)
        self.top_n = self.strategy_config.get('top_n', 20)
        self.max_single_weight = self.risk_config.get('max_single_weight', 0.10)
        self.max_turnover = self.risk_config.get('max_turnover', 0.40)
        self.commission = self.costs_config.get('commission', 0.0003)
        self.slippage = self.costs_config.get('slippage', 0.0002)
        
        # Qlib配置
        self.qlib_initialized = False
        self.qlib_config = self.data_config.get('qlib', {})
        self.provider_uri = self.qlib_config.get('provider_uri', 'D:/qlib_data/qlib_data')
        self.region = self.qlib_config.get('region', 'cn')
        
        # 状态
        self.current_positions = {}  # {symbol: shares}
        self.cash = self.capital
        self.portfolio_value = self.capital
        
        # 记录
        self.trades = []
        self.daily_values = []
        self.positions_history = []
        
        # 初始化Qlib
        self.init_qlib()
    
    def load_predictions(self, date_str):
        """加载预测文件"""
        pred_file = f"data/predictions/pred_{date_str}.csv"
        
        if not os.path.exists(pred_file):
            return None
        
        try:
            predictions = pd.read_csv(pred_file)
            return predictions
        except Exception as e:
            default_logger.warning(f"加载预测文件失败: {e}")
            return None
    
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
    
    def load_prices(self, date_str):
        """
        从Qlib加载指定日期的股价数据
        
        Args:
            date_str: 日期字符串，格式 'YYYY-MM-DD'
        
        Returns:
            dict: {instrument: price} 价格字典，如果失败返回None
        """
        if not self.qlib_initialized:
            self.init_qlib()
        
        try:
            # 将日期字符串转换为Qlib需要的格式
            date_obj = pd.to_datetime(date_str)
            
             # 如果传入的是特殊字符串'csi300_file'，从文件读取
            
            provider_uri = self.data_config['qlib']['provider_uri']
            csi300_file = os.path.join(provider_uri, 'instruments', 'csi300.txt')
            instruments = self.load_instruments_from_file(csi300_file)
            if not instruments:
                default_logger.warning("从文件读取股票代码为空，使用默认值")
                instruments = 'csi300'
            
            
            # 将列表转换为字符串形式（用于日志显示）
            if isinstance(instruments, list):
                default_logger.info(f"使用股票代码列表: 共{len(instruments)}只股票")
                if len(instruments) <= 10:
                    default_logger.info(f"股票代码数组: {instruments}")
                else:
                    # 显示前5个和后5个
                    sample = instruments[:5] + ['...'] + instruments[-5:]
                    default_logger.info(f"股票代码示例: {sample}")

            # 获取该日期的所有股票收盘价
            # 使用前一个交易日的数据（如果当天没有数据）
            # 先尝试获取当天数据
            try:
                data = D.features(
                    instruments=instruments,
                    fields=['$close'],
                    start_time=date_str,
                    end_time=date_str,
                    freq='day'
                )
                default_logger.info(f"股票代码数组data ==== : {data}")

            except:
                # 如果当天没有数据，尝试获取前一个交易日
                prev_date = date_obj - pd.Timedelta(days=1)
                prev_date_str = prev_date.strftime('%Y-%m-%d')
                default_logger.warning(f"日期 {date_str} 无数据，尝试使用前一个交易日: {prev_date_str}")
                try:
                    data = D.features(
                        instruments='all',
                        fields=['$close'],
                        start_time=prev_date_str,
                        end_time=prev_date_str,
                        freq='day'
                    )
                except Exception as e:
                    default_logger.warning(f"无法获取价格数据: {e}")
                    return None
            
            if data.empty:
                default_logger.warning(f"日期 {date_str} 无价格数据")
                return None
            
            # 转换为字典格式 {instrument: price}
            prices = {}
            
            # 处理MultiIndex数据 (datetime, instrument)
            if isinstance(data.index, pd.MultiIndex):
                # 获取指定日期或之前最近的数据
                date_level = data.index.get_level_values('datetime')
                available_dates = date_level[date_level <= date_obj]
                
                if len(available_dates) == 0:
                    # 如果没有该日期或之前的数据，尝试获取最近的数据
                    if len(data) > 0:
                        latest_date = date_level.max()
                        data = data.loc[data.index.get_level_values('datetime') == latest_date]
                    else:
                        return None
                else:
                    # 使用最接近的日期
                    target_date = available_dates.max()
                    data = data.loc[data.index.get_level_values('datetime') == target_date]
                
                # 提取价格数据
                for (dt, instrument), row in data.iterrows():
                    try:
                        # 获取收盘价
                        if isinstance(row, pd.Series):
                            price = row.iloc[0] if len(row) > 0 else None
                        else:
                            price = float(row) if not pd.isna(row) else None
                        
                        if price is not None and not pd.isna(price) and price > 0:
                            # 转换股票代码格式
                            instrument_code = self._normalize_instrument(dt)
                            if instrument_code:
                                prices[instrument_code] = float(price)
                    except Exception as e:
                        default_logger.debug(f"处理 {instrument} 价格失败: {e}")
                        continue
            else:
                # 单层索引，直接处理
                for idx, row in data.iterrows():
                    try:
                        instrument = idx if isinstance(idx, str) else str(idx)
                        if isinstance(row, pd.Series):
                            price = row.iloc[0] if len(row) > 0 else None
                        else:
                            price = float(row) if not pd.isna(row) else None
                        
                        if price is not None and not pd.isna(price) and price > 0:
                            instrument_code = self._normalize_instrument(instrument)
                            if instrument_code:
                                prices[instrument_code] = float(price)
                    except Exception as e:
                        default_logger.debug(f"处理价格数据失败: {e}")
                        continue
            
            if len(prices) == 0:
                default_logger.warning(f"日期 {date_str} 未获取到有效价格数据")
                return None
            
            default_logger.info(f"日期 {date_str} 加载了 {len(prices)} 只股票的价格数据")
            return prices
            
        except Exception as e:
            default_logger.error(f"加载价格数据失败 {date_str}: {e}")
            return None
    


    def load_instruments_from_file(self, file_path):
        """
        从文件中读取股票代码列表
        
        Args:
            file_path: 股票代码文件路径
        
        Returns:
            股票代码列表，格式如 ["600000", "000001", ...] (只保留前6位数字代码)
        """
        if not os.path.exists(file_path):
            default_logger.error(f"股票代码文件不存在: {file_path}")
            return []
        
        instruments = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # 去除空白字符和换行符
                    line = line.strip()
                    if line and not line.startswith('#'):  # 忽略空行和注释
                        # 按TAB分割，只取第一部分（股票代码）
                        # 格式: 000001\t2015-01-05\t2025-10-24
                        code = line.split('\t')[0].strip()
                        
                        # 如果有 SH/SZ 前缀，去掉它
                        if code.startswith('SH') or code.startswith('SZ'):
                            code = code[2:]
                        
                        # 只保留前6位数字
                        if code:
                            instruments.append(code[:6])
            
            default_logger.info(f"从文件加载了 {len(instruments)} 只股票代码: {file_path}")
            default_logger.info(f"股票代码格式: ['000001', '000002', ...] (6位纯数字)")
            return instruments
        except Exception as e:
            default_logger.error(f"读取股票代码文件失败: {e}")
            return []



    def _normalize_instrument(self, instrument):
        """
        标准化股票代码格式
        将各种格式转换为统一格式：600000.SH, 000001.SZ
        
        Args:
            instrument: 股票代码，可能是 '600000', 'SH600000', '600000.SH' 等格式
        
        Returns:
            str: 标准化后的股票代码
        """
        if pd.isna(instrument):
            return None
        
        instrument = str(instrument).strip()
        
        # 如果已经是标准格式，直接返回
        if '.' in instrument:
            return instrument
        
        # 移除前缀
        if instrument.startswith('SH') or instrument.startswith('SZ'):
            instrument = instrument[2:]
        
        # 只保留数字部分（前6位）
        if len(instrument) > 6:
            instrument = instrument[:6]
        
        # 判断市场
        if instrument.startswith('6') or instrument.startswith('9'):
            return f"{instrument}.SH"
        elif instrument.startswith('0') or instrument.startswith('3'):
            return f"{instrument}.SZ"
        else:
            # 默认返回原格式
            return instrument
    
    def select_stocks(self, predictions):
        """选股"""
        predictions = predictions.sort_values('score', ascending=False)
        selected = predictions.head(self.top_n)
        return selected
    
    def calculate_target_weights(self, selected_stocks):
        """计算目标权重（等权重）"""
        n_stocks = len(selected_stocks)
        if n_stocks == 0:
            return {}
        
        weight = min(1.0 / n_stocks, self.max_single_weight)
        
        target_weights = {}
        for _, row in selected_stocks.iterrows():
            target_weights[row['instrument']] = weight
        
        # 归一化
        total = sum(target_weights.values())
        if total > 0:
            for symbol in target_weights:
                target_weights[symbol] /= total
        
        return target_weights
    
    def calculate_turnover(self, current_weights, target_weights):
        """计算换手率"""
        turnover = 0
        
        all_symbols = set(list(current_weights.keys()) + list(target_weights.keys()))
        
        for symbol in all_symbols:
            current_w = current_weights.get(symbol, 0)
            target_w = target_weights.get(symbol, 0)
            turnover += abs(target_w - current_w)
        
        return turnover / 2
    
    def rebalance(self, date_str, prices):
        """调仓"""
        default_logger.info(f"调仓日期: {date_str}")
        
        # 加载预测
        predictions = self.load_predictions(date_str)
        if predictions is None:
            default_logger.warning("无预测数据")
            return
        
        # 选股
        selected = self.select_stocks(predictions)
        if len(selected) == 0:
            default_logger.warning("无选中股票")
            return
        
        # 计算目标权重
        target_weights = self.calculate_target_weights(selected)
        
        # 计算当前权重
        current_weights = self.get_current_weights(prices)
        
        # 检查换手率
        turnover = self.calculate_turnover(current_weights, target_weights)
        default_logger.info(f"换手率: {turnover:.2%}")
        
        if turnover > self.max_turnover:
            default_logger.warning(f"换手率超限，调整权重")
            # 简化处理：等比例缩放
            target_weights = self.adjust_for_turnover(
                current_weights, target_weights, turnover
            )
        
        # 执行交易
        self.execute_trades(target_weights, prices, date_str)
        
        # 记录
        self.positions_history.append({
            'date': date_str,
            'positions': self.current_positions.copy(),
            'weights': target_weights.copy()
        })
    
    def get_current_weights(self, prices):
        """获取当前持仓权重"""
        current_value = {}
        total_value = self.cash
        
        for symbol, shares in self.current_positions.items():
            if symbol in prices:
                value = shares * prices[symbol]
                current_value[symbol] = value
                total_value += value
        
        current_weights = {}
        if total_value > 0:
            for symbol, value in current_value.items():
                current_weights[symbol] = value / total_value
        
        return current_weights
    
    def adjust_for_turnover(self, current_weights, target_weights, turnover):
        """调整权重以满足换手限制"""
        blend_ratio = self.max_turnover / turnover
        
        adjusted_weights = {}
        for symbol in set(list(current_weights.keys()) + list(target_weights.keys())):
            current_w = current_weights.get(symbol, 0)
            target_w = target_weights.get(symbol, 0)
            adjusted_w = current_w + (target_w - current_w) * blend_ratio
            
            if adjusted_w > 0.001:
                adjusted_weights[symbol] = adjusted_w
        
        # 归一化
        total = sum(adjusted_weights.values())
        if total > 0:
            for symbol in adjusted_weights:
                adjusted_weights[symbol] /= total
        
        return adjusted_weights
    
    def execute_trades(self, target_weights, prices, date_str):
        """执行交易"""
        # 计算目标持仓
        target_positions = {}
        total_value = self.portfolio_value
        
        for symbol, weight in target_weights.items():
            if symbol in prices:
                target_value = total_value * weight
                # 考虑滑点和手续费后的价格
                buy_price = prices[symbol] * (1 + self.slippage)
                target_shares = int(target_value / buy_price / 100) * 100
                target_positions[symbol] = target_shares
        
        # 先卖出
        for symbol, current_shares in list(self.current_positions.items()):
            target_shares = target_positions.get(symbol, 0)
            
            if target_shares < current_shares:
                sell_shares = current_shares - target_shares
                sell_price = prices[symbol] * (1 - self.slippage)
                sell_value = sell_shares * sell_price
                cost = sell_value * self.commission
                
                self.cash += sell_value - cost
                self.current_positions[symbol] = target_shares
                
                if target_shares == 0:
                    del self.current_positions[symbol]
                
                self.trades.append({
                    'date': date_str,
                    'symbol': symbol,
                    'action': 'sell',
                    'shares': sell_shares,
                    'price': sell_price,
                    'value': sell_value,
                    'cost': cost
                })
        
        # 再买入
        for symbol, target_shares in target_positions.items():
            current_shares = self.current_positions.get(symbol, 0)
            
            if target_shares > current_shares:
                buy_shares = target_shares - current_shares
                buy_price = prices[symbol] * (1 + self.slippage)
                buy_value = buy_shares * buy_price
                cost = buy_value * self.commission
                
                if self.cash >= buy_value + cost:
                    self.cash -= buy_value + cost
                    self.current_positions[symbol] = target_shares
                    
                    self.trades.append({
                        'date': date_str,
                        'symbol': symbol,
                        'action': 'buy',
                        'shares': buy_shares,
                        'price': buy_price,
                        'value': buy_value,
                        'cost': cost
                    })
    
    def update_portfolio_value(self, date_str, prices):
        """更新组合价值"""
        holdings_value = 0
        
        for symbol, shares in self.current_positions.items():
            if symbol in prices:
                holdings_value += shares * prices[symbol]
        
        self.portfolio_value = self.cash + holdings_value
        
        self.daily_values.append({
            'date': date_str,
            'cash': self.cash,
            'holdings': holdings_value,
            'total': self.portfolio_value
        })
    
    def run(self, start_date, end_date):
        """运行回测"""
        default_logger.info("开始回测")
        default_logger.info(f"初始资金: {self.capital:,.0f}")
        default_logger.info(f"时间范围: {start_date} ~ {end_date}")
        
        # 获取所有周一
        dates = pd.date_range(start=start_date, end=end_date, freq='W-MON')
        
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            
            # 加载价格数据
            prices = self.load_prices(date_str)
            
            if prices is None:
                # 如果没有实际价格数据，使用模拟数据
                default_logger.warning(f"使用模拟价格数据: {date_str}")
                continue
            
            # 调仓
            self.rebalance(date_str, prices)
            
            # 更新组合价值
            self.update_portfolio_value(date_str, prices)
        
        # 计算回测指标
        self.calculate_metrics()
    
    def calculate_metrics(self):
        """计算回测指标"""
        if len(self.daily_values) == 0:
            default_logger.warning("无回测数据")
            return
        
        df = pd.DataFrame(self.daily_values)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # 收益率
        df['return'] = df['total'].pct_change()
        
        # 累计收益
        total_return = (df['total'].iloc[-1] / df['total'].iloc[0]) - 1
        
        # 年化收益
        days = (df.index[-1] - df.index[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1
        
        # 最大回撤
        df['cummax'] = df['total'].cummax()
        df['drawdown'] = (df['total'] - df['cummax']) / df['cummax']
        max_drawdown = df['drawdown'].min()
        
        # 夏普比率
        returns = df['return'].dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(52) if len(returns) > 0 else 0
        
        # 打印结果
        default_logger.info("=" * 60)
        default_logger.info("回测结果:")
        default_logger.info(f"  初始资金: {self.capital:,.0f}")
        default_logger.info(f"  最终资金: {df['total'].iloc[-1]:,.0f}")
        default_logger.info(f"  总收益率: {total_return:.2%}")
        default_logger.info(f"  年化收益: {annual_return:.2%}")
        default_logger.info(f"  最大回撤: {max_drawdown:.2%}")
        default_logger.info(f"  夏普比率: {sharpe_ratio:.2f}")
        default_logger.info(f"  总交易次数: {len(self.trades)}")
        default_logger.info("=" * 60)
        
        # 保存结果
        self.save_results(df)
    
    def save_results(self, df):
        """保存回测结果"""
        output_dir = self.config.get('output', {}).get('report_path', 'data/backtest_results')
        ensure_dir(output_dir)
        
        # 保存权益曲线
        df[['total', 'cash', 'holdings']].to_csv(f"{output_dir}/equity_curve.csv")
        
        # 保存交易记录
        if len(self.trades) > 0:
            pd.DataFrame(self.trades).to_csv(f"{output_dir}/trades.csv", index=False)
        
        # 保存持仓记录
        if len(self.positions_history) > 0:
            positions_df = pd.DataFrame([
                {'date': p['date'], 'symbol': s, 'shares': shares, 'weight': p['weights'].get(s, 0)}
                for p in self.positions_history
                for s, shares in p['positions'].items()
            ])
            positions_df.to_csv(f"{output_dir}/positions.csv", index=False)
        
        default_logger.info(f"回测结果已保存到: {output_dir}")


def main():
    """主函数"""
    # 设置日志
    logger = setup_logger(
        'vnpy_backtest',
        f'logs/vnpy_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    
    # 加载配置
    config = load_yaml('config/backtest.yaml')
    data_config = load_yaml('config/data.yaml')
    
    # 检查是否使用真实的vn.py回测引擎
    use_vnpy_engine = config.get('backtest', {}).get('use_vnpy_engine', False)
    
    if use_vnpy_engine:
        # 使用真实的vn.py回测引擎
        try:
            from backtest.vnpy_backtest_engine import VnpyBacktester, QlibDataAdapter
            
            default_logger.info("使用vn.py真实回测引擎")
            
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
            
            logger.info("vn.py回测完成")
            
        except ImportError as e:
            default_logger.error(f"无法导入vn.py回测引擎: {e}")
            default_logger.info("回退到简单回测引擎")
            use_vnpy_engine = False
    
    if not use_vnpy_engine:
        # 使用简单回测引擎
        default_logger.info("使用简单回测引擎")
        backtester = SimpleBacktester(config, data_config)
        
        # 运行回测
        backtest_config = config.get('backtest', {})
        start_date = backtest_config.get('start_date', '2020-01-01')
        end_date = backtest_config.get('end_date', '2024-12-31')
        
        backtester.run(start_date, end_date)
        
        logger.info("回测完成")


if __name__ == '__main__':
    main()

