"""回测模块"""
from .run_vnpy_backtest import SimpleBacktester

# 尝试导入vnpy相关模块，如果失败则忽略
try:
    from .vnpy_weekly_strategy import WeeklyRotationStrategy
    __all__ = ['WeeklyRotationStrategy', 'SimpleBacktester']
except ImportError:
    # vnpy未安装时，只导出SimpleBacktester
    __all__ = ['SimpleBacktester']

