"""测试预测文件修复效果"""
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import pandas as pd
from pipeline.run_predict import RollingPredictor
from utils.logger import setup_logger, default_logger
from utils.tools import load_yaml

def test_predict():
    """测试单个日期的预测"""
    print("=" * 80)
    print("测试预测修复")
    print("=" * 80)
    
    # 设置日志
    logger = setup_logger('test_predict', 'logs/test_predict_fix.log')
    
    # 加载配置
    data_config = load_yaml('config/data.yaml')
    pipeline_config = load_yaml('config/pipeline.yaml')
    model_config = load_yaml('config/model.yaml')
    
    # 创建预测器
    predictor = RollingPredictor(data_config, pipeline_config, model_config)
    
    # 测试预测一个日期
    predict_date = '2024-08-23'
    model_path = f'data/models/model_{predict_date.replace("-", "")}.txt'
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先运行训练或指定一个存在的日期")
        return
    
    print(f"\n开始预测: {predict_date}")
    print(f"模型路径: {model_path}")
    
    try:
        result = predictor.predict_single_date(predict_date, model_path, 'data')
        
        if result is not None:
            print(f"\n✅ 预测成功！")
            print(f"预测结果行数: {len(result)}")
            print(f"\n前10条预测结果:")
            print(result.head(10))
            
            # 验证预测文件
            pred_file = f'data/predictions/pred_{predict_date}.csv'
            if os.path.exists(pred_file):
                df = pd.read_csv(pred_file)
                print(f"\n预测文件验证:")
                print(f"  文件路径: {pred_file}")
                print(f"  总行数: {len(df)}")
                print(f"  唯一股票数: {df['instrument'].nunique()}")
                print(f"  唯一日期数: {df['date'].nunique()}")
                
                if len(df) <= 350:  # 允许一些误差
                    print(f"  ✅ 文件格式正确！每只股票一个预测")
                else:
                    print(f"  ⚠️ 文件可能包含多个日期的数据")
        else:
            print("❌ 预测失败")
            
    except Exception as e:
        print(f"❌ 预测出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_predict()

