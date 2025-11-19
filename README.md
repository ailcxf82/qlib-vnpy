# Qlib + vn.py 量化交易系统

一个完整的量化交易系统，集成了Qlib因子计算、机器学习模型训练和vn.py回测功能。

## 项目简介

本项目实现了一个完整的量化交易工作流：

1. **Qlib模块**：负责因子计算、特征工程和模型训练
   - 技术指标因子（MA、RSI、MACD、布林带等）
   - 基本面因子（PE、PB、ROE、盈利能力等）
   - 周频滚动训练（3年训练窗口，每周向前滚动）
   - 支持LightGBM、XGBoost、CatBoost等模型
   - GPU加速支持

2. **vn.py模块**：负责策略回测和实盘交易
   - 周频调仓策略
   - 费用控制（佣金0.03%，滑点0.02%）
   - 风险控制（最大单股权重10%，换手率限制40%）
   - 完整的回测报告（收益、回撤、夏普比率）

## 项目结构

```
quant_system/
├── config/                 # 配置文件
│   ├── data.yaml          # 数据配置
│   ├── factor.yaml        # 因子配置
│   ├── model.yaml         # 模型配置
│   ├── pipeline.yaml      # 训练流程配置
│   └── backtest.yaml      # 回测配置
│
├── data/                   # 数据目录
│   ├── raw/               # 原始数据
│   ├── intermediate/      # 中间数据
│   ├── predictions/       # 预测结果
│   └── models/            # 保存的模型
│
├── factors/                # 因子计算
│   ├── base_factors.py    # 技术因子
│   ├── fundamental_factors.py  # 基本面因子
│   └── factor_engine.py   # 因子引擎
│
├── feature/                # 特征工程
│   ├── label.py           # 标签生成
│   ├── dataset_builder.py # 数据集构建
│   └── feature_pipeline.py # 特征流程
│
├── model/                  # 模型训练
│   ├── lgb_model.py       # LightGBM模型
│   ├── trainer.py         # 训练器
│   ├── predictor.py       # 预测器
│   └── metrics.py         # 评估指标
│
├── pipeline/               # 训练流程
│   ├── run_train.py       # 滚动训练
│   ├── run_predict.py     # 滚动预测
│   └── run_all.py         # 一键运行
│
├── backtest/               # 回测模块
│   ├── vnpy_weekly_strategy.py  # vn.py策略
│   └── run_vnpy_backtest.py     # 回测引擎
│
├── utils/                  # 工具模块
│   ├── logger.py          # 日志工具
│   ├── timer.py           # 计时工具
│   └── tools.py           # 通用工具
│
├── requirements.txt        # 依赖包
└── README.md              # 本文件
```

## 安装

### 1. 克隆项目

```bash
git clone <repository_url>
cd Qlib1114
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 初始化Qlib数据

```bash
# 下载中国A股数据（示例）
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

或者使用自定义数据路径，修改 `config/data.yaml` 中的 `provider_uri`。

## 使用方法

### 1. 修改配置

根据实际情况修改配置文件：

- `config/data.yaml`：数据路径、股票池、时间范围
- `config/factor.yaml`：因子列表
- `config/model.yaml`：模型参数
- `config/pipeline.yaml`：训练参数
- `config/backtest.yaml`：回测参数

### 2. 滚动训练

```bash
python pipeline/run_train.py
```

这将执行周频滚动训练，训练窗口为3年，每周向前滚动1周。训练好的模型将保存在 `data/models/` 目录。

### 3. 滚动预测

```bash
python pipeline/run_predict.py
```

这将使用训练好的模型对每周进行预测，预测结果保存在 `data/predictions/` 目录，格式为 `pred_2020-01-06.csv`。

### 4. 一键运行（训练+预测）

```bash
python pipeline/run_all.py
```

### 5. 运行回测

```bash
python backtest/run_vnpy_backtest.py
```

回测结果将保存在 `data/backtest_results/` 目录，包括：
- `equity_curve.csv`：权益曲线
- `trades.csv`：交易记录
- `positions.csv`：持仓记录

## 功能特性

### Qlib模块

#### 1. 技术因子

- **移动平均线**：MA5, MA10, MA20, MA60
- **动量指标**：RSI6, RSI12, RSI24, ROC10, ROC20
- **波动率**：Volatility10, Volatility20, Volatility60
- **趋势指标**：MACD, MACD Signal, MACD Histogram
- **通道指标**：布林带上轨、中轨、下轨
- **波幅指标**：ATR14
- **成交量指标**：Volume MA, Volume Ratio, OBV
- **摆动指标**：KDJ(K, D, J)

#### 2. 基本面因子

- **估值因子**：PE（市盈率）, PB（市净率）, PS（市销率）
- **盈利能力**：ROE（净资产收益率）, ROA（总资产收益率）, ProfitMargin（净利润率）
- **偿债能力**：DebtRatio（资产负债率）, CurrentRatio（流动比率）
- **成长性**：RevenueGrowth（营收增长率）, NetProfitGrowth（净利润增长率）
- **规模因子**：MarketCap（市值）, LogMarketCap（对数市值）

#### 3. 模型训练

- **支持的模型**：LightGBM（默认）, XGBoost, CatBoost
- **训练方式**：周频滚动训练
- **训练窗口**：3年（156周）
- **滚动步长**：1周
- **特征预处理**：缩尾、标准化、填充缺失值
- **GPU加速**：支持LightGBM GPU加速

#### 4. 标签生成

- **下周收益率**：预测未来5个交易日（1周）的收益率
- **支持的标签类型**：回归标签、排名标签、二分类标签

### vn.py回测模块

#### 1. 策略逻辑

- **调仓频率**：每周一
- **选股方式**：按预测分数排序，选择Top N（默认20只）
- **权重分配**：等权重
- **持仓限制**：单股最大权重10%

#### 2. 风险控制

- **换手率限制**：单次调仓不超过40%
- **权重调整**：自动调整权重以满足换手限制
- **行业中性**：可配置（可选功能）

#### 3. 成本设置

- **佣金**：0.03%（双边）
- **滑点**：0.02%（买入加，卖出减）

#### 4. 回测输出

- **收益指标**：总收益率、年化收益率
- **风险指标**：最大回撤、夏普比率
- **交易统计**：总交易次数、换手率
- **可视化**：权益曲线、持仓记录

## GPU加速

### 启用LightGBM GPU加速

1. 安装CUDA（需要NVIDIA GPU）

2. 安装GPU版本的LightGBM：

```bash
pip install lightgbm --install-option=--gpu
```

3. 修改 `config/model.yaml`：

```yaml
lightgbm:
  device: "gpu"
  gpu_platform_id: 0
  gpu_device_id: 0
```

## 评估指标

- **IC**：信息系数（Pearson相关）
- **Rank IC**：排名信息系数（Spearman相关）
- **MSE**：均方误差
- **MAE**：平均绝对误差
- **多空收益**：Top N多头 vs Bottom N空头的收益差

## 扩展功能

### 切换模型

修改 `config/model.yaml` 中的 `model_type`：

```yaml
model_type: "xgboost"  # 或 "catboost"
```

### 添加自定义因子

在 `factors/base_factors.py` 或 `factors/fundamental_factors.py` 中添加新的因子计算方法。

### 修改选股逻辑

修改 `backtest/vnpy_weekly_strategy.py` 中的 `select_stocks` 和 `calculate_target_weights` 方法。

## 注意事项

1. **数据准备**：确保Qlib数据已正确初始化
2. **时间范围**：训练窗口需要足够的历史数据（至少3年）
3. **内存占用**：处理大量股票和因子时可能占用较多内存
4. **回测局限**：简化回测引擎不包含完整的市场微观结构模拟
5. **实盘注意**：实盘交易前请充分测试，注意流动性和冲击成本

## 日志

所有运行日志保存在 `logs/` 目录：

- `rolling_train_*.log`：训练日志
- `rolling_predict_*.log`：预测日志
- `vnpy_backtest_*.log`：回测日志

## 性能优化建议

1. **并行计算**：在因子计算中使用多进程
2. **数据缓存**：缓存中间结果避免重复计算
3. **特征选择**：使用特征重要度进行特征筛选
4. **增量训练**：对于新数据可以使用增量训练

## 常见问题

### Q: Qlib初始化失败？
A: 检查数据路径是否正确，确保已下载数据。

### Q: 模型训练很慢？
A: 考虑使用GPU加速，或减少特征数量、调整模型参数。

### Q: 预测文件找不到？
A: 确保先运行训练和预测流程，生成预测文件。

### Q: 回测结果不理想？
A: 检查因子有效性、模型参数、选股逻辑等，进行优化。

## 联系方式

如有问题或建议，请提交Issue或Pull Request。

## 许可证

本项目采用MIT许可证。

---

**免责声明**：本项目仅供学习研究使用，不构成任何投资建议。实盘交易风险自负。

