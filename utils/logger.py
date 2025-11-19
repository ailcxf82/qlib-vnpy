"""日志工具"""
import logging
import os
from datetime import datetime


def setup_logger(name, log_file=None, level=logging.INFO):
    """设置日志器"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# 默认日志器
default_logger = setup_logger(
    'quant_system',
    f'logs/system_{datetime.now().strftime("%Y%m%d")}.log'
)

