"""计时工具"""
import time
from functools import wraps
from utils.logger import default_logger


class Timer:
    """计时器上下文管理器"""
    
    def __init__(self, name=""):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        if self.name:
            default_logger.info(f"{self.name} 耗时: {elapsed:.2f}秒")
        else:
            default_logger.info(f"耗时: {elapsed:.2f}秒")


def timing_decorator(func):
    """函数计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with Timer(f"{func.__name__}"):
            return func(*args, **kwargs)
    return wrapper

