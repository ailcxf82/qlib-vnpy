"""测试回测数据加载功能"""
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import pandas as pd
import sys
# 直接导入，避免通过__init__.py导入vnpy相关模块
sys.path.insert(0, str(Path(__file__).parent))
from run_vnpy_backtest import SimpleBacktester
from utils.logger import setup_logger
from utils.tools import load_yaml
from datetime import datetime


def test_qlib_initialization():
    """测试Qlib初始化"""
    print("=" * 60)
    print("测试1: Qlib初始化")
    print("=" * 60)
    
    try:
        config = load_yaml('config/backtest.yaml')
        data_config = load_yaml('config/data.yaml')
        
        backtester = SimpleBacktester(config, data_config)
        print("✓ Qlib初始化成功")
        print(f"  数据路径: {backtester.provider_uri}")
        print(f"  区域: {backtester.region}")
        return True
    except Exception as e:
        print(f"✗ Qlib初始化失败: {e}")
        return False


def test_load_prices():
    """测试价格数据加载"""
    print("\n" + "=" * 60)
    print("测试2: 价格数据加载")
    print("=" * 60)
    
    try:
        config = load_yaml('config/backtest.yaml')
        data_config = load_yaml('config/data.yaml')
        
        backtester = SimpleBacktester(config, data_config)
        
        # 测试几个日期
        test_dates = ['2020-01-06', '2020-01-13', '2020-01-20']
        
        for date_str in test_dates:
            print(f"\n测试日期: {date_str}")
            prices = backtester.load_prices(date_str)
            
            if prices is None:
                print(f"  ✗ 无法加载价格数据")
                continue
            
            print(f"  ✓ 成功加载 {len(prices)} 只股票的价格")
            
            # 显示前5只股票的价格
            sample_count = min(5, len(prices))
            sample_items = list(prices.items())[:sample_count]
            for instrument, price in sample_items:
                print(f"    {instrument}: {price:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ 价格数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_instrument_normalization():
    """测试股票代码标准化"""
    print("\n" + "=" * 60)
    print("测试3: 股票代码标准化")
    print("=" * 60)
    
    try:
        config = load_yaml('config/backtest.yaml')
        data_config = load_yaml('config/data.yaml')
        
        backtester = SimpleBacktester(config, data_config)
        
        test_cases = [
            ('600000', '600000.SH'),
            ('000001', '000001.SZ'),
            ('SH600000', '600000.SH'),
            ('SZ000001', '000001.SZ'),
            ('600000.SH', '600000.SH'),
            ('000001.SZ', '000001.SZ'),
        ]
        
        all_passed = True
        for input_code, expected in test_cases:
            result = backtester._normalize_instrument(input_code)
            if result == expected:
                print(f"  ✓ {input_code} -> {result}")
            else:
                print(f"  ✗ {input_code} -> {result} (期望: {expected})")
                all_passed = False
        
        return all_passed
    except Exception as e:
        print(f"✗ 股票代码标准化测试失败: {e}")
        return False


def test_backtest_run():
    """测试回测运行（小范围）"""
    print("\n" + "=" * 60)
    print("测试4: 回测运行（小范围测试）")
    print("=" * 60)
    
    try:
        config = load_yaml('config/backtest.yaml')
        data_config = load_yaml('config/data.yaml')
        
        backtester = SimpleBacktester(config, data_config)
        
        # 只测试一周的数据
        start_date = '2020-01-06'
        end_date = '2020-01-13'
        
        print(f"回测时间范围: {start_date} ~ {end_date}")
        print("开始运行回测...")
        
        backtester.run(start_date, end_date)
        
        print("\n✓ 回测运行完成")
        print(f"  交易次数: {len(backtester.trades)}")
        print(f"  记录天数: {len(backtester.daily_values)}")
        
        if len(backtester.daily_values) > 0:
            final_value = backtester.daily_values[-1]['total']
            print(f"  最终资金: {final_value:,.0f}")
        
        return True
    except Exception as e:
        print(f"✗ 回测运行测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("回测数据加载功能验证")
    print("=" * 60)
    
    results = []
    
    # 测试1: Qlib初始化
    results.append(("Qlib初始化", test_qlib_initialization()))
    
    # 测试2: 价格数据加载
    results.append(("价格数据加载", test_load_prices()))
    
    # 测试3: 股票代码标准化
    results.append(("股票代码标准化", test_instrument_normalization()))
    
    # 测试4: 回测运行
    results.append(("回测运行", test_backtest_run()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n✓ 所有测试通过！回测数据加载功能正常。")
    else:
        print("\n✗ 部分测试失败，请检查错误信息。")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

