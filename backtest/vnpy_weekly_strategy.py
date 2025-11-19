"""vn.py周频调仓策略"""
import pandas as pd
import numpy as np
from datetime import datetime
from vnpy.app.cta_strategy import CtaTemplate, StopOrder
from vnpy.trader.object import BarData, TickData, OrderData, TradeData
from vnpy.trader.constant import Direction, Offset, Interval
import os


class WeeklyRotationStrategy(CtaTemplate):
    """周频轮动策略"""
    
    author = "Qlib + vn.py"
    
    # 策略参数
    top_n = 20  # 选股数量
    max_single_weight = 0.10  # 单股最大权重
    max_turnover = 0.40  # 最大换手率
    commission_rate = 0.0003  # 佣金
    slippage_rate = 0.0002  # 滑点
    
    # 策略变量
    prediction_dir = "data/predictions"  # 预测文件目录
    current_positions = {}  # 当前持仓 {symbol: weight}
    target_positions = {}  # 目标持仓
    is_monday = False  # 是否周一
    
    parameters = [
        "top_n",
        "max_single_weight",
        "max_turnover",
        "commission_rate",
        "slippage_rate"
    ]
    
    variables = [
        "current_positions",
        "target_positions",
        "is_monday"
    ]
    
    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """初始化"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        
        self.last_rebalance_date = None
        self.current_date = None
        self.prices = {}  # 最新价格
        self.capital = 10000000  # 初始资金
        self.current_value = self.capital
        
    def on_init(self):
        """策略初始化"""
        self.write_log("策略初始化")
        
        # 加载历史数据
        self.load_bar(10)
    
    def on_start(self):
        """策略启动"""
        self.write_log("策略启动")
    
    def on_stop(self):
        """策略停止"""
        self.write_log("策略停止")
    
    def on_bar(self, bar: BarData):
        """K线更新"""
        # 更新价格
        self.prices[bar.vt_symbol] = bar.close_price
        
        # 更新日期
        current_date = bar.datetime.date()
        
        if self.current_date != current_date:
            self.current_date = current_date
            
            # 判断是否周一
            if bar.datetime.weekday() == 0:  # 周一
                self.is_monday = True
                self.write_log(f"周一调仓日: {current_date}")
                self.rebalance()
            else:
                self.is_monday = False
    
    def load_predictions(self, date):
        """
        加载预测文件
        
        Args:
            date: 日期
        
        Returns:
            预测结果DataFrame
        """
        date_str = date.strftime("%Y-%m-%d")
        pred_file = f"{self.prediction_dir}/pred_{date_str}.csv"
        
        if not os.path.exists(pred_file):
            self.write_log(f"预测文件不存在: {pred_file}")
            return None
        
        try:
            predictions = pd.read_csv(pred_file)
            self.write_log(f"加载预测文件: {pred_file}, 共 {len(predictions)} 只股票")
            return predictions
        except Exception as e:
            self.write_log(f"加载预测文件失败: {e}")
            return None
    
    def select_stocks(self, predictions):
        """
        选股
        
        Args:
            predictions: 预测结果
        
        Returns:
            选中的股票列表
        """
        # 按分数排序，选择top_n
        predictions = predictions.sort_values('score', ascending=False)
        selected = predictions.head(self.top_n)
        
        self.write_log(f"选股完成: {len(selected)} 只股票")
        
        return selected
    
    def calculate_target_weights(self, selected_stocks):
        """
        计算目标权重
        
        Args:
            selected_stocks: 选中的股票
        
        Returns:
            目标权重字典 {symbol: weight}
        """
        # 等权重
        n_stocks = len(selected_stocks)
        equal_weight = 1.0 / n_stocks if n_stocks > 0 else 0
        
        # 限制单股最大权重
        weight = min(equal_weight, self.max_single_weight)
        
        target_weights = {}
        total_weight = 0
        
        for _, row in selected_stocks.iterrows():
            symbol = row['instrument']
            target_weights[symbol] = weight
            total_weight += weight
        
        # 归一化
        if total_weight > 0:
            for symbol in target_weights:
                target_weights[symbol] /= total_weight
        
        self.write_log(f"目标权重计算完成: {len(target_weights)} 只股票")
        
        return target_weights
    
    def calculate_turnover(self, target_weights):
        """
        计算换手率
        
        Args:
            target_weights: 目标权重
        
        Returns:
            换手率
        """
        turnover = 0
        
        # 卖出的权重
        for symbol, current_weight in self.current_positions.items():
            if symbol not in target_weights:
                turnover += abs(current_weight)
            else:
                turnover += abs(target_weights[symbol] - current_weight)
        
        # 买入的权重
        for symbol, target_weight in target_weights.items():
            if symbol not in self.current_positions:
                turnover += abs(target_weight)
        
        turnover = turnover / 2  # 换手率定义为买卖双边之和的一半
        
        return turnover
    
    def adjust_weights_for_turnover(self, target_weights):
        """
        根据换手限制调整权重
        
        Args:
            target_weights: 目标权重
        
        Returns:
            调整后的权重
        """
        turnover = self.calculate_turnover(target_weights)
        
        if turnover <= self.max_turnover:
            return target_weights
        
        self.write_log(f"换手率 {turnover:.2%} 超过限制 {self.max_turnover:.2%}，调整权重")
        
        # 简单实现：保持部分旧持仓，减少换手
        adjusted_weights = {}
        
        # 混合新旧权重
        blend_ratio = self.max_turnover / turnover
        
        for symbol in set(list(self.current_positions.keys()) + list(target_weights.keys())):
            current_w = self.current_positions.get(symbol, 0)
            target_w = target_weights.get(symbol, 0)
            
            adjusted_w = current_w + (target_w - current_w) * blend_ratio
            
            if adjusted_w > 0.001:  # 过滤很小的权重
                adjusted_weights[symbol] = adjusted_w
        
        # 归一化
        total = sum(adjusted_weights.values())
        if total > 0:
            for symbol in adjusted_weights:
                adjusted_weights[symbol] /= total
        
        return adjusted_weights
    
    def rebalance(self):
        """调仓"""
        if self.current_date is None:
            return
        
        # 1. 加载预测
        predictions = self.load_predictions(self.current_date)
        
        if predictions is None:
            self.write_log("无预测数据，跳过调仓")
            return
        
        # 2. 选股
        selected_stocks = self.select_stocks(predictions)
        
        if len(selected_stocks) == 0:
            self.write_log("无选中股票，跳过调仓")
            return
        
        # 3. 计算目标权重
        target_weights = self.calculate_target_weights(selected_stocks)
        
        # 4. 换手控制
        target_weights = self.adjust_weights_for_turnover(target_weights)
        
        # 5. 执行交易
        self.execute_trades(target_weights)
        
        # 6. 更新持仓
        self.current_positions = target_weights.copy()
        self.last_rebalance_date = self.current_date
        
        self.write_log(f"调仓完成: {len(self.current_positions)} 只股票")
    
    def execute_trades(self, target_weights):
        """
        执行交易
        
        Args:
            target_weights: 目标权重
        """
        # 计算目标持仓量
        target_positions = {}
        
        for symbol, weight in target_weights.items():
            if symbol not in self.prices:
                continue
            
            price = self.prices[symbol]
            target_value = self.current_value * weight
            target_volume = int(target_value / price / 100) * 100  # 转换为手（100股）
            
            if target_volume > 0:
                target_positions[symbol] = target_volume
        
        # 当前持仓量
        current_positions_volume = {}
        for symbol, weight in self.current_positions.items():
            if symbol in self.prices:
                price = self.prices[symbol]
                volume = int(self.current_value * weight / price / 100) * 100
                current_positions_volume[symbol] = volume
        
        # 执行交易
        # 先卖出
        for symbol, current_volume in current_positions_volume.items():
            target_volume = target_positions.get(symbol, 0)
            
            if target_volume < current_volume:
                sell_volume = current_volume - target_volume
                self.sell(symbol, self.prices[symbol], sell_volume)
                self.write_log(f"卖出: {symbol}, 价格: {self.prices[symbol]}, 数量: {sell_volume}")
        
        # 再买入
        for symbol, target_volume in target_positions.items():
            current_volume = current_positions_volume.get(symbol, 0)
            
            if target_volume > current_volume:
                buy_volume = target_volume - current_volume
                self.buy(symbol, self.prices[symbol], buy_volume)
                self.write_log(f"买入: {symbol}, 价格: {self.prices[symbol]}, 数量: {buy_volume}")
    
    def on_order(self, order: OrderData):
        """订单更新"""
        pass
    
    def on_trade(self, trade: TradeData):
        """成交更新"""
        self.write_log(f"成交: {trade.vt_symbol}, 方向: {trade.direction}, "
                       f"价格: {trade.price}, 数量: {trade.volume}")
    
    def on_stop_order(self, stop_order: StopOrder):
        """停止单更新"""
        pass

