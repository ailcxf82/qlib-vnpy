"""基本面因子计算"""
import pandas as pd
import numpy as np


class FundamentalFactors:
    """基本面因子"""
    
    @staticmethod
    def calculate_pe(data):
        """市盈率 = 市值 / 净利润"""
        if 'pe_ratio' in data.columns:
            return data['pe_ratio']
        if 'market_cap' in data.columns and 'net_profit' in data.columns:
            return data['market_cap'] / (data['net_profit'] + 1e-10)
        return pd.Series(np.nan, index=data.index)
    
    @staticmethod
    def calculate_pb(data):
        """市净率 = 市值 / 净资产"""
        if 'pb_ratio' in data.columns:
            return data['pb_ratio']
        if 'market_cap' in data.columns and 'net_assets' in data.columns:
            return data['market_cap'] / (data['net_assets'] + 1e-10)
        return pd.Series(np.nan, index=data.index)
    
    @staticmethod
    def calculate_ps(data):
        """市销率 = 市值 / 营收"""
        if 'ps_ratio' in data.columns:
            return data['ps_ratio']
        if 'market_cap' in data.columns and 'revenue' in data.columns:
            return data['market_cap'] / (data['revenue'] + 1e-10)
        return pd.Series(np.nan, index=data.index)
    
    @staticmethod
    def calculate_roe(data):
        """净资产收益率 = 净利润 / 净资产"""
        if 'roe' in data.columns:
            return data['roe']
        if 'net_profit' in data.columns and 'net_assets' in data.columns:
            return data['net_profit'] / (data['net_assets'] + 1e-10)
        return pd.Series(np.nan, index=data.index)
    
    @staticmethod
    def calculate_roa(data):
        """总资产收益率 = 净利润 / 总资产"""
        if 'roa' in data.columns:
            return data['roa']
        if 'net_profit' in data.columns and 'total_assets' in data.columns:
            return data['net_profit'] / (data['total_assets'] + 1e-10)
        return pd.Series(np.nan, index=data.index)
    
    @staticmethod
    def calculate_profit_margin(data):
        """净利润率 = 净利润 / 营收"""
        if 'profit_margin' in data.columns:
            return data['profit_margin']
        if 'net_profit' in data.columns and 'revenue' in data.columns:
            return data['net_profit'] / (data['revenue'] + 1e-10)
        return pd.Series(np.nan, index=data.index)
    
    @staticmethod
    def calculate_debt_ratio(data):
        """资产负债率 = 总负债 / 总资产"""
        if 'debt_ratio' in data.columns:
            return data['debt_ratio']
        if 'total_liabilities' in data.columns and 'total_assets' in data.columns:
            return data['total_liabilities'] / (data['total_assets'] + 1e-10)
        return pd.Series(np.nan, index=data.index)
    
    @staticmethod
    def calculate_current_ratio(data):
        """流动比率 = 流动资产 / 流动负债"""
        if 'current_ratio' in data.columns:
            return data['current_ratio']
        if 'current_assets' in data.columns and 'current_liabilities' in data.columns:
            return data['current_assets'] / (data['current_liabilities'] + 1e-10)
        return pd.Series(np.nan, index=data.index)
    
    @staticmethod
    def calculate_revenue_growth(data, period=4):
        """营收增长率"""
        if 'revenue' in data.columns:
            return data['revenue'].pct_change(period)
        return pd.Series(np.nan, index=data.index)
    
    @staticmethod
    def calculate_profit_growth(data, period=4):
        """净利润增长率"""
        if 'net_profit' in data.columns:
            return data['net_profit'].pct_change(period)
        return pd.Series(np.nan, index=data.index)
    
    @staticmethod
    def calculate_market_cap(data):
        """市值"""
        if 'market_cap' in data.columns:
            return data['market_cap']
        if 'close' in data.columns and 'total_shares' in data.columns:
            return data['close'] * data['total_shares']
        return pd.Series(np.nan, index=data.index)
    
    def calculate_all(self, data):
        """计算所有基本面因子"""
        factors = pd.DataFrame(index=data.index)
        
        # 估值因子
        factors['PE'] = self.calculate_pe(data)
        factors['PB'] = self.calculate_pb(data)
        factors['PS'] = self.calculate_ps(data)
        
        # 盈利能力
        factors['ROE'] = self.calculate_roe(data)
        factors['ROA'] = self.calculate_roa(data)
        factors['ProfitMargin'] = self.calculate_profit_margin(data)
        
        # 偿债能力
        factors['DebtRatio'] = self.calculate_debt_ratio(data)
        factors['CurrentRatio'] = self.calculate_current_ratio(data)
        
        # 成长性
        factors['RevenueGrowth'] = self.calculate_revenue_growth(data)
        factors['NetProfitGrowth'] = self.calculate_profit_growth(data)
        
        # 市值
        factors['MarketCap'] = self.calculate_market_cap(data)
        factors['LogMarketCap'] = np.log(factors['MarketCap'] + 1)
        
        return factors

