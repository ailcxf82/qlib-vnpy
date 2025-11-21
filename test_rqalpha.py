"""
测试 RQAlpha 回测
快速验证脚本
"""
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from backtest.rqalpha_backtest import run_rqalpha_backtest

if __name__ == '__main__':
    print("=" * 80)
    print("开始 RQAlpha 回测测试")
    print("=" * 80)
    
    # 检查是否安装 RQAlpha
    try:
        import rqalpha
        print(f"✓ RQAlpha 版本: {rqalpha.__version__}")
    except ImportError:
        print("✗ 未安装 RQAlpha")
        print("\n请安装 RQAlpha:")
        print("  pip install rqalpha")
        print("  rqalpha update_bundle  # 下载数据")
        sys.exit(1)
     
    # 检查预测文件
    pred_dir = 'data/predictions'
    if os.path.exists(pred_dir):
        pred_files = [f for f in os.listdir(pred_dir) if f.startswith('pred_') and f.endswith('.csv')]
        print(f"✓ 找到 {len(pred_files)} 个预测文件")
    else:
        print(f"✗ 预测目录不存在: {pred_dir}")
        sys.exit(1)
    
    print("\n开始回测...")
    print("=" * 80)
    
    # 运行回测
    report = run_rqalpha_backtest()
    
    if report:
        print("\n✓ 回测成功完成！")
        print("\n查看结果文件:")
        print("  - data/backtest_results/rqalpha_portfolio.csv  (权益曲线)")
        print("  - data/backtest_results/rqalpha_trades.csv     (交易明细)")
        print("  - data/backtest_results/rqalpha_positions.csv  (持仓记录)")
        print("  - data/backtest_results/rqalpha_summary.csv    (回测摘要)")
    else:
        print("\n✗ 回测失败，请查看日志文件")

