"""因子计算引擎"""
import warnings
# 过滤 joblib resource_tracker 的警告（Windows系统常见问题，不影响功能）
# 使用多种方式确保过滤所有相关警告
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')
warnings.filterwarnings('ignore', message='.*resource_tracker.*')
warnings.filterwarnings('ignore', message='.*FileNotFoundError.*系统找不到指定的路径.*')

import qlib
import pandas as pd
import os
from qlib.data import D
from factors.base_factors import TechnicalFactors
from factors.fundamental_factors import FundamentalFactors
from utils.logger import default_logger
from utils.timer import timing_decorator
from utils.tools import ensure_dir


class FactorEngine:
    """因子计算引擎"""
    
    def __init__(self, config):
        self.config = config
        self.tech_factors = TechnicalFactors()
        self.fund_factors = FundamentalFactors()
        self.qlib_initialized = False
    
    def init_qlib(self):
        """初始化Qlib"""
        if not self.qlib_initialized:
            provider_uri = self.config['qlib']['provider_uri']
            region = self.config['qlib']['region']
            
            try:
                qlib.init(provider_uri=provider_uri, region=region)
                self.qlib_initialized = True
                default_logger.info(f"Qlib初始化成功: {provider_uri}")
            except Exception as e:
                default_logger.error(f"Qlib初始化失败: {e}")
                raise
    
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
    
    @timing_decorator
    def fetch_qlib_data(self, instruments, start_time, end_time, fields=None):
        """从Qlib获取数据"""
        self.init_qlib()
        
        if fields is None:
            fields = ['$open', '$high', '$low', '$close', '$volume', '$factor']
        
        # 如果传入的是特殊字符串'csi300_file'，从文件读取
        if isinstance(instruments, str) and instruments == 'csi300_file':
            provider_uri = self.config['qlib']['provider_uri']
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
        
        try:
            data = D.features(
                instruments=instruments,
                # instruments=["600000", "000001", "000063", "000157", "000166"],   # 测试用
                fields=fields,
                start_time=start_time,
                end_time=end_time,
                freq='day'
            )
            default_logger.info(f"成功获取数据: {len(data)} 条记录")
            return data
        except Exception as e:
            default_logger.error(f"获取数据失败: {e}")
            raise





    def calculate_factors_for_stock(self, stock_data):
        """计算单只股票的因子"""
        # 技术因子
        tech_factors = self.tech_factors.calculate_all(stock_data)
        
        # 基本面因子
        fund_factors = self.fund_factors.calculate_all(stock_data)
        
        # 合并
        all_factors = pd.concat([tech_factors, fund_factors], axis=1)
        
        return all_factors
    
    @timing_decorator
    def calculate_factors(self, start_time, end_time, instruments='csi300'):
        """计算所有股票的因子"""
        default_logger.info(f"开始计算因子: {start_time} to {end_time}")
        
        # 获取原始数据
        raw_data = self.fetch_qlib_data(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time
        )
        
        # 为每只股票计算因子
        all_factors_list = []
        
        for instrument in raw_data.index.get_level_values(0).unique():
            stock_data = raw_data.loc[instrument]
            stock_data.columns = [col.replace('$', '') for col in stock_data.columns]
            
            try:
                factors = self.calculate_factors_for_stock(stock_data)
                factors['instrument'] = instrument
                all_factors_list.append(factors)
            except Exception as e:
                default_logger.warning(f"计算 {instrument} 因子失败: {e}")
                continue
        
        # 合并所有股票的因子
        if all_factors_list:
            all_factors = pd.concat(all_factors_list)
            default_logger.info(f"因子计算完成: {all_factors.shape}")
            return all_factors
        else:
            default_logger.error("没有成功计算任何因子")
            return None
    
    def save_factors(self, factors, output_path):
        """保存因子数据"""
        ensure_dir(output_path)
        factors.to_pickle(output_path)
        default_logger.info(f"因子数据已保存: {output_path}")
    
    def load_factors(self, input_path):
        """加载因子数据"""
        factors = pd.read_pickle(input_path)
        default_logger.info(f"因子数据已加载: {input_path}")
        return factors

