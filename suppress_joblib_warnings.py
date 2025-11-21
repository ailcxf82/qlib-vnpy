"""
专门用于抑制 joblib resource_tracker 警告的模块
必须在所有其他导入之前导入此模块
"""
import warnings
import re
import sys

# 保存原始的 showwarning 函数
_original_showwarning = warnings.showwarning

def _filtered_showwarning(message, category, filename, lineno, file=None, line=None):
    """自定义的警告显示函数，过滤 joblib resource_tracker 警告"""
    # 检查是否是 joblib 相关的警告
    if category == UserWarning:
        msg_str = str(message)
        filename_str = str(filename) if filename else ''
        
        # 检查是否是 resource_tracker 相关的警告
        if ('resource_tracker' in msg_str.lower() or 
            'resource_tracker' in filename_str.lower() or
            'joblib' in filename_str.lower() and 'loky' in filename_str.lower() or
            '系统找不到指定的路径' in msg_str or
            'FileNotFoundError' in msg_str and 'joblib_memmapping_folder' in msg_str):
            # 直接返回，不显示警告
            return
    
    # 对于其他警告，使用原始函数显示
    _original_showwarning(message, category, filename, lineno, file, line)

# 替换警告显示函数
warnings.showwarning = _filtered_showwarning

# 同时使用标准的过滤方法（双重保险）
try:
    warnings.filterwarnings('ignore', category=UserWarning, module='joblib')
    warnings.filterwarnings('ignore', category=UserWarning, module='joblib.externals')
    warnings.filterwarnings('ignore', category=UserWarning, module='joblib.externals.loky')
    warnings.filterwarnings('ignore', category=UserWarning, module='joblib.externals.loky.backend')
    warnings.filterwarnings('ignore', category=UserWarning, module='joblib.externals.loky.backend.resource_tracker')
except:
    pass

# 按消息内容过滤（使用字符串匹配，因为 warnings.filterwarnings 不支持正则表达式对象）
try:
    warnings.filterwarnings('ignore', category=UserWarning, message='resource_tracker')
    warnings.filterwarnings('ignore', category=UserWarning, message='FileNotFoundError')
    # 注意：中文字符串匹配可能不够精确，主要依赖 _filtered_showwarning 函数
except:
    pass

