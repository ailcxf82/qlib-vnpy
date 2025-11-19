"""通用工具函数"""
import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path


def load_yaml(file_path):
    """加载YAML配置文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(data, file_path):
    """保存YAML配置文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True)


def ensure_dir(path):
    """确保目录存在（支持文件路径和目录路径）"""
    path_obj = Path(path)
    # 如果是文件路径（有扩展名），则获取其父目录
    if path_obj.suffix:
        path_obj = path_obj.parent
    # 创建目录
    path_obj.mkdir(parents=True, exist_ok=True)
    return str(path_obj)


def winsorize(series, quantile_range=(0.01, 0.99)):
    """缩尾处理"""
    lower = series.quantile(quantile_range[0])
    upper = series.quantile(quantile_range[1])
    return series.clip(lower, upper)


def standardize(series):
    """标准化"""
    return (series - series.mean()) / (series.std() + 1e-8)


def neutralize(factor, group_by=None):
    """中性化处理"""
    if group_by is None:
        return factor - factor.mean()
    else:
        return factor.groupby(group_by).transform(lambda x: x - x.mean())


def get_trade_dates(start_date, end_date, freq='W-MON'):
    """获取交易日期列表（周频）"""
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    return [d.strftime('%Y-%m-%d') for d in dates]


def get_real_trade_dates(start_date, end_date, calendar_file='D:/qlib_data/qlib_data/calendars/day.txt'):
    """
    从Qlib日历文件中获取真实交易日期
    
    Args:
        start_date: 开始日期，格式 'YYYY-MM-DD'
        end_date: 结束日期，格式 'YYYY-MM-DD'
        calendar_file: 交易日历文件路径，默认为Qlib的day.txt
    
    Returns:
        list: 区间内的交易日期列表，格式 ['YYYY-MM-DD', ...]
    """
    if not os.path.exists(calendar_file):
        raise FileNotFoundError(f"交易日历文件不存在: {calendar_file}")
    
    # 读取交易日历文件
    trade_dates = []
    try:
        with open(calendar_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # 文件格式通常是 'YYYY-MM-DD' 或时间戳
                    try:
                        # 尝试解析日期
                        date_obj = pd.to_datetime(line.split()[0])
                        trade_dates.append(date_obj)
                    except:
                        continue
    except Exception as e:
        raise RuntimeError(f"读取交易日历文件失败: {e}")
    
    if not trade_dates:
        raise ValueError("交易日历文件为空或格式不正确")
    
    # 转换为DataFrame便于筛选
    df = pd.DataFrame({'date': trade_dates})
    df['date'] = pd.to_datetime(df['date'])
    
    # 筛选区间内的交易日
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    mask = (df['date'] >= start) & (df['date'] <= end)
    filtered_dates = df.loc[mask, 'date']
    
    # 转换为字符串列表
    result = [d.strftime('%Y-%m-%d') for d in filtered_dates]
    
    return result


def get_real_trade_dates_weekly(start_date, end_date, weekday=0, calendar_file='D:/qlib_data/qlib_data/calendars/day.txt'):
    """
    从Qlib日历文件中获取真实交易日期，并筛选出指定星期几
    
    Args:
        start_date: 开始日期，格式 'YYYY-MM-DD'
        end_date: 结束日期，格式 'YYYY-MM-DD'
        weekday: 星期几，0=周一, 1=周二, ..., 6=周日
        calendar_file: 交易日历文件路径
    
    Returns:
        list: 区间内指定星期几的交易日期列表
    """
    # 获取所有交易日
    all_trade_dates = get_real_trade_dates(start_date, end_date, calendar_file)
    
    # 筛选出指定星期几
    result = []
    for date_str in all_trade_dates:
        date_obj = pd.to_datetime(date_str)
        if date_obj.weekday() == weekday:
            result.append(date_str)
    
    return result

