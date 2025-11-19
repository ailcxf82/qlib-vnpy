"""技术因子计算"""
import pandas as pd
import numpy as np


class TechnicalFactors:
    """技术指标因子"""
    
    @staticmethod
    def calculate_ma(data, period):
        """移动平均线"""
        return data['close'].rolling(window=period).mean()
    
    @staticmethod
    def calculate_rsi(data, period=14):
        """相对强弱指标"""
        close = data['close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_volatility(data, period=20):
        """波动率"""
        return data['close'].pct_change().rolling(window=period).std()
    
    @staticmethod
    def calculate_macd(data, fast=12, slow=26, signal=9):
        """MACD指标"""
        close = data['close']
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(data, period=20, std_dev=2):
        """布林带"""
        close = data['close']
        ma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = ma + (std_dev * std)
        lower = ma - (std_dev * std)
        return upper, ma, lower
    
    @staticmethod
    def calculate_atr(data, period=14):
        """平均真实波幅"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def calculate_roc(data, period=20):
        """变动率指标"""
        close = data['close']
        return (close / close.shift(period) - 1) * 100
    
    @staticmethod
    def calculate_volume_ma(data, period=5):
        """成交量移动平均"""
        return data['volume'].rolling(window=period).mean()
    
    @staticmethod
    def calculate_obv(data):
        """能量潮指标"""
        close = data['close']
        volume = data['volume']
        direction = np.where(close.diff() > 0, 1, -1)
        direction[0] = 0
        obv = (volume * direction).cumsum()
        return obv
    
    @staticmethod
    def calculate_kdj(data, period=9):
        """KDJ指标"""
        low_min = data['low'].rolling(window=period).min()
        high_max = data['high'].rolling(window=period).max()
        
        rsv = (data['close'] - low_min) / (high_max - low_min + 1e-10) * 100
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d
        
        return k, d, j
    
    def calculate_all(self, data):
        """计算所有技术因子"""
        factors = pd.DataFrame(index=data.index)
        
        # 移动平均线
        factors['MA5'] = self.calculate_ma(data, 5)
        factors['MA10'] = self.calculate_ma(data, 10)
        factors['MA20'] = self.calculate_ma(data, 20)
        factors['MA60'] = self.calculate_ma(data, 60)
        
        # RSI
        factors['RSI6'] = self.calculate_rsi(data, 6)
        factors['RSI12'] = self.calculate_rsi(data, 12)
        factors['RSI24'] = self.calculate_rsi(data, 24)
        
        # 波动率
        factors['Volatility10'] = self.calculate_volatility(data, 10)
        factors['Volatility20'] = self.calculate_volatility(data, 20)
        factors['Volatility60'] = self.calculate_volatility(data, 60)
        
        # MACD
        macd, signal, hist = self.calculate_macd(data)
        factors['MACD'] = macd
        factors['MACD_Signal'] = signal
        factors['MACD_Hist'] = hist
        
        # 布林带
        upper, middle, lower = self.calculate_bollinger_bands(data)
        factors['BOLL_Upper'] = upper
        factors['BOLL_Middle'] = middle
        factors['BOLL_Lower'] = lower
        factors['BOLL_Width'] = (upper - lower) / middle
        
        # ATR
        factors['ATR14'] = self.calculate_atr(data, 14)
        
        # ROC
        factors['ROC10'] = self.calculate_roc(data, 10)
        factors['ROC20'] = self.calculate_roc(data, 20)
        
        # 成交量指标
        factors['Volume_MA5'] = self.calculate_volume_ma(data, 5)
        factors['Volume_MA10'] = self.calculate_volume_ma(data, 10)
        factors['Volume_Ratio'] = data['volume'] / (factors['Volume_MA5'] + 1e-10)
        
        # OBV
        factors['OBV'] = self.calculate_obv(data)
        
        # KDJ
        k, d, j = self.calculate_kdj(data)
        factors['KDJ_K'] = k
        factors['KDJ_D'] = d
        factors['KDJ_J'] = j
        
        # 价格动量
        factors['Return_1d'] = data['close'].pct_change(1)
        factors['Return_5d'] = data['close'].pct_change(5)
        factors['Return_20d'] = data['close'].pct_change(20)
        
        return factors

