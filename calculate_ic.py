"""计算和展示模型IC值"""
# 必须在所有其他导入之前抑制 joblib 警告
import suppress_joblib_warnings  # noqa: F401

import warnings
import os

# 额外的警告过滤（双重保险）
# 注意：warnings.filterwarnings 的 message 参数只接受字符串，不支持正则表达式
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')
warnings.filterwarnings('ignore', category=UserWarning, message='resource_tracker')
warnings.filterwarnings('ignore', category=UserWarning, message='FileNotFoundError')

import pandas as pd
import numpy as np
from glob import glob
from model.metrics import Metrics
from scipy.stats import pearsonr, spearmanr

print("="*80)
print("模型IC值分析报告")
print("="*80)

# 1. 查找预测文件和对应的真实收益率
pred_files = sorted(glob('data/predictions/pred_*.csv'))
print(f"\n找到 {len(pred_files)} 个预测文件")

if len(pred_files) == 0:
    print("\n⚠️ 未找到预测文件！")
    print("请先运行预测：python pipeline/run_rolling_predict.py")
    exit()

# 2. 读取预测结果和计算真实收益率
print("\n正在计算IC值...")

ic_results = []
rank_ic_results = []
dates = []

# 初始化Qlib（用于获取真实收益率）
try:
    import qlib
    from qlib.data import D
    
    # 检查配置文件
    import yaml
    with open('config/data.yaml', 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    provider_uri = data_config['qlib']['provider_uri']
    if not os.path.exists(provider_uri):
        raise FileNotFoundError(f"Qlib数据目录不存在: {provider_uri}")
    
    qlib.init(provider_uri=provider_uri, region='cn')
    print("✓ Qlib初始化成功")
    
    # 读取配置中的预测周期
    with open('config/pipeline.yaml', 'r', encoding='utf-8') as f:
        pipeline_config = yaml.safe_load(f)
    forward_days = pipeline_config.get('label', {}).get('forward_days', 5)
    print(f"✓ 预测周期: {forward_days}天")
    
except Exception as e:
    print(f"\n⚠️ 无法初始化Qlib: {e}")
    print("将使用简化方法估算IC（基于历史数据）")
    qlib_available = False
else:
    qlib_available = True

# 3. 对每个预测文件计算IC
print(f"\n开始计算IC值（使用最近50个预测文件）...\n")

# 只取最近的50个文件
recent_files = pred_files[-50:] if len(pred_files) > 50 else pred_files

for i, pred_file in enumerate(recent_files, 1):
    try:
        # 提取日期
        date_str = os.path.basename(pred_file).replace('pred_', '').replace('.csv', '')
        
        # 读取预测结果
        pred_df = pd.read_csv(pred_file)
        
        if len(pred_df) < 5:  # 样本太少，跳过
            continue
        
        if qlib_available:
            # 计算该日期之后的真实收益率
            try:
                # 获取股票列表
                instruments = pred_df['instrument'].tolist()
                
                # 获取价格数据
                end_date = pd.Timestamp(date_str) + pd.Timedelta(days=forward_days+5)
                price_data = D.features(
                    instruments=instruments,
                    fields=['$close'],
                    start_time=date_str,
                    end_time=end_date.strftime('%Y-%m-%d'),
                    freq='day'
                )
                
                # 计算收益率
                true_returns = []
                for inst in instruments:
                    try:
                        inst_prices = price_data.loc[inst]['close']
                        if len(inst_prices) >= forward_days + 1:
                            start_price = inst_prices.iloc[0]
                            end_price = inst_prices.iloc[forward_days]
                            ret = (end_price - start_price) / start_price
                            true_returns.append(ret)
                        else:
                            true_returns.append(np.nan)
                    except:
                        true_returns.append(np.nan)
                
                pred_df['true_return'] = true_returns
                
                # 移除NaN
                valid_data = pred_df.dropna(subset=['score', 'true_return'])
                
                if len(valid_data) >= 10:  # 至少10个有效样本
                    # 计算IC (Pearson相关系数)
                    ic, _ = pearsonr(valid_data['score'], valid_data['true_return'])
                    
                    # 计算Rank IC (Spearman相关系数)
                    rank_ic, _ = spearmanr(valid_data['score'], valid_data['true_return'])
                    
                    ic_results.append(ic)
                    rank_ic_results.append(rank_ic)
                    dates.append(date_str)
                    
                    if i % 10 == 0:
                        print(f"  进度: {i}/{len(recent_files)} - {date_str}: IC={ic:.4f}, RankIC={rank_ic:.4f}")
            
            except Exception as e:
                # print(f"  跳过 {date_str}: {e}")
                pass
        
    except Exception as e:
        print(f"  错误处理 {pred_file}: {e}")
        continue

# 4. 展示结果
if len(ic_results) > 0:
    print("\n" + "="*80)
    print("IC统计结果")
    print("="*80)
    
    ic_array = np.array(ic_results)
    rank_ic_array = np.array(rank_ic_results)
    
    print(f"\n【IC (Pearson相关系数)】")
    print(f"  样本数量: {len(ic_results)}")
    print(f"  平均IC: {np.mean(ic_array):.4f}")
    print(f"  中位数IC: {np.median(ic_array):.4f}")
    print(f"  标准差: {np.std(ic_array):.4f}")
    print(f"  最大值: {np.max(ic_array):.4f}")
    print(f"  最小值: {np.min(ic_array):.4f}")
    print(f"  IC>0的比例: {(ic_array > 0).sum() / len(ic_array) * 100:.1f}%")
    print(f"  IC>0.02的比例: {(ic_array > 0.02).sum() / len(ic_array) * 100:.1f}%")
    
    print(f"\n【Rank IC (Spearman相关系数)】")
    print(f"  平均Rank IC: {np.mean(rank_ic_array):.4f}")
    print(f"  中位数Rank IC: {np.median(rank_ic_array):.4f}")
    print(f"  标准差: {np.std(rank_ic_array):.4f}")
    print(f"  最大值: {np.max(rank_ic_array):.4f}")
    print(f"  最小值: {np.min(rank_ic_array):.4f}")
    print(f"  RankIC>0的比例: {(rank_ic_array > 0).sum() / len(rank_ic_array) * 100:.1f}%")
    
    # IC稳定性指标
    ic_ir = np.mean(ic_array) / (np.std(ic_array) + 1e-8)  # IC信息比率
    print(f"\n【IC稳定性】")
    print(f"  IC信息比率 (IC_IR): {ic_ir:.4f}")
    print(f"  说明: IC_IR = 平均IC / IC标准差，衡量IC的稳定性")
    if ic_ir > 1.0:
        print(f"  评价: ✓ 优秀 (>1.0)")
    elif ic_ir > 0.5:
        print(f"  评价: ✓ 良好 (0.5-1.0)")
    else:
        print(f"  评价: ⚠ 需要改进 (<0.5)")
    
    # 展示最近10次IC
    print(f"\n【最近10次IC值】")
    print(f"  日期          IC      Rank IC")
    print(f"  " + "-"*35)
    for i in range(min(10, len(dates))):
        idx = -(i+1)
        print(f"  {dates[idx]}  {ic_results[idx]:+.4f}  {rank_ic_results[idx]:+.4f}")
    
    # IC分布统计
    print(f"\n【IC分布】")
    bins = [(-1, -0.05), (-0.05, 0), (0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 1)]
    for low, high in bins:
        count = ((ic_array >= low) & (ic_array < high)).sum()
        pct = count / len(ic_array) * 100
        bar = '█' * int(pct / 2)
        print(f"  [{low:+.2f}, {high:+.2f}): {count:3d} ({pct:5.1f}%) {bar}")
    
    # 保存IC序列
    ic_df = pd.DataFrame({
        'date': dates,
        'IC': ic_results,
        'Rank_IC': rank_ic_results
    })
    ic_df.to_csv('data/ic_analysis.csv', index=False)
    print(f"\n✓ IC序列已保存到: data/ic_analysis.csv")
    
    # 性能评价
    print("\n" + "="*80)
    print("模型性能评价")
    print("="*80)
    
    avg_ic = np.mean(ic_array)
    avg_rank_ic = np.mean(rank_ic_array)
    
    print(f"\n基于平均IC={avg_ic:.4f}的评价：")
    if avg_ic > 0.05:
        print(f"  ⭐⭐⭐⭐⭐ 优秀！IC>0.05，模型预测能力很强")
    elif avg_ic > 0.03:
        print(f"  ⭐⭐⭐⭐ 良好！IC>0.03，模型有较好的预测能力")
    elif avg_ic > 0.02:
        print(f"  ⭐⭐⭐ 中等。IC>0.02，模型有一定预测能力")
    elif avg_ic > 0:
        print(f"  ⭐⭐ 较弱。IC>0但<0.02，模型预测能力有限")
    else:
        print(f"  ⭐ 需要改进。IC<0，模型可能存在问题")
    
    print(f"\n建议：")
    if avg_ic < 0.03:
        print(f"  1. 检查标签类型（建议使用 excess_return）")
        print(f"  2. 检查特征标准化方法（建议使用 rank）")
        print(f"  3. 增加更多有区分度的因子")
        print(f"  4. 调整模型参数（树深度、学习率等）")
    else:
        print(f"  ✓ 模型表现良好，可以继续优化特征工程和参数调优")
    
else:
    print("\n⚠️ 无法计算IC值")
    print("可能原因：")
    print("  1. 预测文件中没有足够的数据")
    print("  2. 无法获取真实收益率数据")
    print("  3. 数据日期不匹配")

print("\n" + "="*80)
print("分析完成")
print("="*80)


