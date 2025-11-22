# 预测信号重复问题诊断报告

## 问题描述

在 `pred_2025-01-22.csv` 中发现：
- **总股票数**: 299只
- **唯一预测值**: 仅30个（唯一性只有10%）
- **最常见值**: 0.009231471183224621 出现了90次（占30.1%）
- **预测标准差**: 只有0.002618，非常小

这意味着**大量不同的股票被赋予了完全相同的预测分数**，这是一个严重的模型问题。

## 根本原因分析

根据代码审查和数据分析，问题的根本原因有以下几点：

### 1. **特征数据质量问题** ⭐⭐⭐⭐⭐（最可能）

从 `feature/dataset_builder.py` 中的预处理流程看：

```python
# 第61-67行：缩尾处理
if self.feature_engineering_config.get('winsorize', True):
    quantile_range = self.feature_engineering_config.get(
        'winsorize_quantile', [0.01, 0.99]
    )
    for col in features.columns:
        if features[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            features[col] = winsorize(features[col], quantile_range)

# 第69-73行：标准化
if self.feature_engineering_config.get('standardize', True):
    for col in features.columns:
        if features[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            features[col] = standardize(features[col])

# 第76行：填充NaN为0
features = features.fillna(0)

# 第79行：替换inf为0
features = features.replace([np.inf, -np.inf], 0)
```

**问题点**：
- 缩尾处理会将极端值压缩到相同的边界值
- 标准化后，如果很多特征都缺失或无效，会被填充为0
- 这导致**大量股票的特征向量变得非常相似甚至完全相同**

### 2. **特征标准化的方法问题** ⭐⭐⭐⭐

当前的标准化方法是**按列(特征)标准化**，即对每个特征计算全局均值和标准差。

**问题**：
- 如果某个特征在所有股票中变化不大，标准化后所有股票在该特征上的值会非常接近
- 应该使用**截面标准化**（cross-sectional standardization），即在每个时间点对所有股票进行标准化

### 3. **基本面因子数据缺失** ⭐⭐⭐⭐

从 `factors/fundamental_factors.py` 看，基本面因子（如PE、PB、ROE等）需要特定的数据字段：

```python
def calculate_pe(data):
    if 'pe_ratio' in data.columns:
        return data['pe_ratio']
    if 'market_cap' in data.columns and 'net_profit' in data.columns:
        return data['market_cap'] / (data['net_profit'] + 1e-10)
    return pd.Series(np.nan, index=data.index)  # ← 很多情况下返回NaN
```

**问题**：
- Qlib基础数据可能不包含这些基本面字段
- 大量基本面因子被填充为NaN，然后被填充为0或均值
- 导致**所有股票在这些特征上完全相同**

### 4. **技术因子同质化** ⭐⭐⭐

从 `factors/base_factors.py` 看，技术因子计算时：

```python
# 计算RSI
factors['RSI6'] = self.calculate_rsi(data, 6)
# 计算MA
factors['MA5'] = self.calculate_ma(data, 5)
```

**问题**：
- 如果数据窗口不足（如新股或停牌），这些指标会产生NaN
- MA、RSI等指标对于走势相似的股票会产生相似的值
- 标准化后差异被进一步压缩

### 5. **模型泛化能力不足** ⭐⭐⭐

LightGBM模型可能过拟合或欠拟合：

**可能原因**：
- 训练样本不足
- 模型参数设置不当（如树深度太浅）
- 特征信息量不足，模型学到的都是噪音

### 6. **预测时使用的是单个时间点的数据** ⭐⭐

```python
# 预测时只有当天的特征
scores = self.trainer.predict(features)
```

**问题**：
- 如果当天很多股票因停牌、数据缺失等原因特征相同
- 模型会给出相同的预测

## 诊断结果汇总

| 问题 | 严重程度 | 影响 |
|------|---------|------|
| 特征数据大量缺失/相同 | 🔴 严重 | 直接导致预测相同 |
| 标准化方法不当 | 🔴 严重 | 压缩特征差异 |
| 基本面因子无效 | 🟠 较严重 | 减少有效特征 |
| 技术因子同质化 | 🟠 较严重 | 特征区分度低 |
| 模型问题 | 🟡 中等 | 无法捕捉细微差异 |

## 建议的解决方案

### 短期解决方案（立即实施）

1. **检查特征数据质量**
   ```python
   # 添加特征质量检查
   - 统计每个特征的唯一值数量
   - 检查特征的标准差
   - 识别常数特征并移除
   ```

2. **改进缺失值处理**
   ```python
   # 不要全部填充为0或均值
   - 对技术指标用前向填充
   - 对基本面指标可以用行业中位数填充
   - 考虑将缺失情况作为一个特征
   ```

3. **使用截面标准化**
   ```python
   # 改为按时间截面标准化
   def standardize_cross_sectional(features):
       if isinstance(features.index, pd.MultiIndex):
           # 对每个日期分别标准化
           return features.groupby(level=1).transform(
               lambda x: (x - x.mean()) / (x.std() + 1e-8)
           )
   ```

4. **移除无效的基本面因子**
   ```python
   # 如果基本面数据不可用，暂时不使用这些因子
   # 专注于技术因子和量价因子
   ```

### 中期解决方案（1-2周）

1. **增加更多有区分度的因子**
   - 相对强度因子（与市场/行业比较）
   - 动量因子（不同周期的收益率）
   - 流动性因子（成交额排名）
   - 个股特有因子（换手率、振幅）

2. **使用排名转换**
   ```python
   # 将因子值转换为排名
   features_rank = features.rank(pct=True)  # 转换为百分位排名
   ```

3. **特征选择**
   - 移除低方差特征
   - 移除高度相关的冗余特征
   - 使用特征重要度筛选

4. **模型调优**
   - 增加树的深度
   - 调整学习率
   - 使用更多的boosting轮数

### 长期解决方案（1-2月）

1. **改进数据源**
   - 补充更完整的基本面数据
   - 使用更高质量的行情数据

2. **构建因子库**
   - Alpha101因子
   - Alpha191因子
   - WorldQuant因子

3. **集成学习**
   - 训练多个模型
   - 使用不同的特征子集
   - 模型预测加权平均

4. **在线学习**
   - 定期更新模型
   - 使用最新数据重训练

## 下一步行动

1. ✅ **立即执行**：检查最近一次预测使用的特征数据
2. ✅ **优先级高**：实施截面标准化
3. ⭐ **优先级高**：移除或修复无效的基本面因子
4. ⭐ **优先级中**：增加更多区分度高的因子
5. 📊 **持续监控**：每次预测后检查唯一值比例

---

## 附录：快速验证脚本

```python
# 检查特征质量
import pandas as pd

# 读取某一天的预测输入特征
features = pd.read_pickle('data/features/features_2025-01-22.pkl')

# 检查特征唯一性
for col in features.columns:
    unique_ratio = features[col].nunique() / len(features)
    print(f"{col}: {unique_ratio:.2%} 唯一")
    if unique_ratio < 0.1:
        print(f"  ⚠️ 警告：{col} 唯一值过少！")

# 检查是否存在完全相同的行
duplicates = features.duplicated().sum()
print(f"\n完全相同的特征行数: {duplicates}/{len(features)}")
```

---
**报告生成时间**: 2025-11-21
**报告作者**: AI Assistant



