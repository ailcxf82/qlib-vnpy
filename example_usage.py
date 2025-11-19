"""使用示例"""
from utils.tools import load_yaml
from utils.logger import setup_logger
import os


def check_environment():
    """检查环境配置"""
    print("=" * 60)
    print("检查环境配置")
    print("=" * 60)
    
    # 检查配置文件
    configs = [
        'config/data.yaml',
        'config/factor.yaml',
        'config/model.yaml',
        'config/pipeline.yaml',
        'config/backtest.yaml'
    ]
    
    print("\n1. 配置文件:")
    for config in configs:
        if os.path.exists(config):
            print(f"  ✓ {config}")
        else:
            print(f"  ✗ {config} (缺失)")
    
    # 检查目录
    dirs = [
        'data',
        'logs',
        'factors',
        'feature',
        'model',
        'pipeline',
        'backtest',
        'utils'
    ]
    
    print("\n2. 目录结构:")
    for d in dirs:
        if os.path.exists(d):
            print(f"  ✓ {d}/")
        else:
            print(f"  ✗ {d}/ (缺失)")
    
    # 检查Python包
    print("\n3. Python包:")
    packages = [
        ('qlib', 'Qlib'),
        ('lightgbm', 'LightGBM'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('sklearn', 'Scikit-learn'),
        ('yaml', 'PyYAML')
    ]
    
    for pkg_name, display_name in packages:
        try:
            __import__(pkg_name)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ✗ {display_name} (未安装)")
    
    print("\n" + "=" * 60)
    print("环境检查完成")
    print("=" * 60)


def example_1_factor_calculation():
    """示例1：因子计算"""
    print("\n" + "=" * 60)
    print("示例1：因子计算")
    print("=" * 60)
    
    from factors.factor_engine import FactorEngine
    
    # 加载配置
    data_config = load_yaml('config/data.yaml')
    
    # 创建因子引擎
    factor_engine = FactorEngine(data_config)
    
    print("\n初始化Qlib...")
    try:
        factor_engine.init_qlib()
        print("✓ Qlib初始化成功")
    except Exception as e:
        print(f"✗ Qlib初始化失败: {e}")
        print("\n请确保:")
        print("1. 已安装Qlib: pip install qlib")
        print("2. 已下载数据: python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn")
        print("3. 配置文件中的数据路径正确")
        return
    
    print("\n计算因子（示例：2024-01-01 到 2024-01-31）...")
    print("注意：这可能需要几分钟时间...")
    
    # 这里只是示例，实际运行需要确保数据存在
    # factors = factor_engine.calculate_factors(
    #     start_time='2024-01-01',
    #     end_time='2024-01-31',
    #     instruments='csi300'
    # )
    
    print("提示：实际运行请取消注释上面的代码")


def example_2_model_training():
    """示例2：模型训练"""
    print("\n" + "=" * 60)
    print("示例2：模型训练")
    print("=" * 60)
    
    print("\n运行滚动训练:")
    print("  python pipeline/run_train.py")
    
    print("\n训练参数（在config/pipeline.yaml中配置）:")
    print("  - 训练窗口: 156周（3年）")
    print("  - 滚动步长: 1周")
    print("  - 标签: 下周收益率")
    
    print("\n训练好的模型将保存在:")
    print("  data/models/model_YYYYMMDD.txt")


def example_3_prediction():
    """示例3：预测"""
    print("\n" + "=" * 60)
    print("示例3：预测")
    print("=" * 60)
    
    print("\n运行滚动预测:")
    print("  python pipeline/run_predict.py")
    
    print("\n预测结果将保存在:")
    print("  data/predictions/pred_YYYY-MM-DD.csv")
    
    print("\n预测文件格式:")
    print("  instrument,score")
    print("  000001.SZ,0.0523")
    print("  000002.SZ,0.0312")
    print("  ...")


def example_4_backtest():
    """示例4：回测"""
    print("\n" + "=" * 60)
    print("示例4：回测")
    print("=" * 60)
    
    print("\n运行回测:")
    print("  python backtest/run_vnpy_backtest.py")
    
    print("\n回测参数（在config/backtest.yaml中配置）:")
    print("  - 初始资金: 1000万")
    print("  - 调仓频率: 每周一")
    print("  - 选股数量: 20只")
    print("  - 佣金: 0.03%")
    print("  - 滑点: 0.02%")
    print("  - 最大换手: 40%")
    print("  - 单股最大权重: 10%")
    
    print("\n回测结果将保存在:")
    print("  data/backtest_results/equity_curve.csv")
    print("  data/backtest_results/trades.csv")
    print("  data/backtest_results/positions.csv")


def example_5_full_pipeline():
    """示例5：完整流程"""
    print("\n" + "=" * 60)
    print("示例5：完整流程")
    print("=" * 60)
    
    print("\n一键运行训练+预测:")
    print("  python pipeline/run_all.py")
    
    print("\n完整工作流程:")
    print("  1. 初始化Qlib数据")
    print("  2. 计算技术因子和基本面因子")
    print("  3. 生成标签（下周收益率）")
    print("  4. 特征预处理（缩尾、标准化）")
    print("  5. 滚动训练LightGBM模型")
    print("  6. 生成预测结果")
    print("  7. vn.py回测")
    print("  8. 输出回测报告")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("Qlib + vn.py 量化交易系统 - 使用示例")
    print("=" * 60)
    
    # 检查环境
    check_environment()
    
    # 示例
    example_1_factor_calculation()
    example_2_model_training()
    example_3_prediction()
    example_4_backtest()
    example_5_full_pipeline()
    
    print("\n" + "=" * 60)
    print("快速开始")
    print("=" * 60)
    print("\n1. 安装依赖:")
    print("   pip install -r requirements.txt")
    
    print("\n2. 初始化Qlib数据:")
    print("   python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn")
    
    print("\n3. 修改配置:")
    print("   编辑 config/*.yaml 文件，根据实际情况调整参数")
    
    print("\n4. 运行完整流程:")
    print("   python pipeline/run_all.py")
    
    print("\n5. 查看结果:")
    print("   - 日志: logs/")
    print("   - 模型: data/models/")
    print("   - 预测: data/predictions/")
    print("   - 回测: data/backtest_results/")
    
    print("\n" + "=" * 60)
    print("更多信息请查看 README.md")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()

