# RQAlpha 回测使用说明

## 概述

本项目集成了 [RQAlpha](https://rqalpha.readthedocs.io/) 回测框架，用于对 Qlib 模型预测结果进行专业的量化回测。

RQAlpha 是由米筐科技开发的开源量化回测框架，具有以下优势：
- ✅ **专业性**：金融行业标准的回测引擎
- ✅ **完整性**：内置交易成本、滑点、撮合机制
- ✅ **可靠性**：经过大量实盘验证
- ✅ **易用性**：API 简洁，文档完善

参考文档：https://rqalpha.readthedocs.io/zh-cn/latest/notebooks/run-rqalpha-in-ipython.html

---

## 安装

### 1. 安装 RQAlpha

```bash
pip install rqalpha
```

### 2. 下载数据

```bash
# 下载 A 股数据（首次使用需要）
rqalpha update_bundle

# 或指定数据路径
rqalpha update_bundle -d ~/.rqalpha
```

---

## 使用方法

### 方式一：直接运行回测

```bash
cd D:\lianghuatouzi\Qlib1114

# 运行回测
python backtest/rqalpha_backtest.py
```

### 方式二：使用测试脚本

```bash
# 快速测试
python test_rqalpha.py
```

---

## 策略逻辑

### 核心流程

```python
1. 初始化
   ↓
2. 每周一调仓
   ├─ 读取预测文件 (data/predictions/pred_YYYY-MM-DD.csv)
   ├─ 按分数排序，选择 Top N 只股票
   ├─ 卖出不在目标列表的股票
   └─ 买入目标股票（等权重配置）
   ↓
3. 收盘后记录持仓
   ↓
4. 生成回测报告
```

### 选股逻辑

```python
# 1. 加载预测文件
predictions = pd.read_csv(f'data/predictions/pred_{date}.csv')

# 2. 按分数排序
predictions = predictions.sort_values('score', ascending=False)

# 3. 选择 Top N
selected = predictions.head(top_n)

# 4. 等权重配置
target_weight = 1.0 / top_n
for stock in selected:
    order_target_percent(stock, target_weight)
```

---

## 配置参数

### config/backtest.yaml

```yaml
backtest:
  start_date: "2024-08-23"  # 回测开始日期
  end_date: "2024-11-01"    # 回测结束日期
  capital: 10000000         # 初始资金（1000万）

strategy:
  top_n: 10                 # 选股数量（Top 10）

costs:
  commission: 0.0003        # 佣金 0.03%
  slippage: 0.0002          # 滑点 0.02%
```

---

## 输出结果

### 1. 控制台输出

```
================================================================================
RQAlpha 回测结果摘要
================================================================================

总收益率: 5.23%
年化收益率: 28.45%
基准收益率: 3.12%
Alpha: 0.0234
Beta: 0.8567
夏普比率: 1.4523
最大回撤: -4.23%
波动率: 12.34%

交易统计:
总交易次数: 120
胜率: 65.00%
================================================================================
```

### 2. 交易明细

```
================================================================================
交易明细（前20条）
================================================================================
                    commission  exec_id  last_price  last_quantity order_book_id  ...
2024-08-26 15:00:00     189.45  1001      15.23       12400       000001.XSHE    ...
2024-08-26 15:00:00     234.67  1002      23.47       10000       600000.XSHG    ...
...
```

### 3. 持仓快照

```
当日收盘持仓 (2024-08-26):
股票代码         数量      成本价      现价        市值         盈亏
--------------------------------------------------------------------------------
000001.XSHE     12400      15.23     15.45    189,780.00     2,728.00
600000.XSHG     10000      23.47     23.89    238,900.00     4,200.00
...
--------------------------------------------------------------------------------
合计                                            4,856,230.00    45,670.00
现金: 5,143,770.00
总资产: 10,000,000.00
================================================================================
```

### 4. 保存的文件

```
data/backtest_results/
├── rqalpha_portfolio.csv   # 权益曲线（日期、现金、持仓市值、总资产）
├── rqalpha_trades.csv      # 交易记录（时间、股票、方向、数量、价格、手续费）
├── rqalpha_positions.csv   # 持仓记录（日期、股票、数量、成本价、现价）
└── rqalpha_summary.csv     # 回测摘要（收益率、回撤、夏普等）
```

---

## 日志记录

### 日志文件位置

```
logs/rqalpha_backtest_YYYYMMDD_HHMMSS.log
```

### 日志内容

```
2024-11-20 10:00:00 ================================================================================
2024-11-20 10:00:00 RQAlpha 回测初始化完成
2024-11-20 10:00:00 选股数量: 10
2024-11-20 10:00:00 ================================================================================

2024-11-20 10:00:05 ================================================================================
2024-11-20 10:00:05 调仓日期: 2024-08-26 (Monday)
2024-11-20 10:00:05 预测股票数: 299
2024-11-20 10:00:05 选中股票数: 10
2024-11-20 10:00:05 
2024-11-20 10:00:05 Top 10 股票:
2024-11-20 10:00:05   000001.XSHE: 得分 0.003685
2024-11-20 10:00:05   600000.XSHG: 得分 0.003373
...
```

---

## 与 SimpleBacktester 对比

| 特性 | RQAlpha | SimpleBacktester |
|------|---------|------------------|
| **专业性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **数据完整性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **撮合机制** | 多种模式 | 简单撮合 |
| **性能指标** | 30+ 指标 | 基础指标 |
| **可视化** | 内置图表 | 需自己实现 |
| **学习曲线** | 中等 | 简单 |
| **依赖** | rqalpha | 纯 Python |

---

## 高级功能

### 1. 自定义撮合类型

```python
config = {
    "base": {
        "matching_type": "next_bar",  # 下一根K线价格成交（避免未来函数）
    }
}
```

### 2. 风险控制

```python
config = {
    "mod": {
        "sys_risk": {
            "enabled": True,
            "max_position_size": 0.15,  # 单只股票最大仓位 15%
        }
    }
}
```

### 3. 绘制回测图表

```python
config = {
    "mod": {
        "sys_analyser": {
            "plot": True,  # 自动绘图
            "plot_save_file": "backtest_chart.png",
        }
    }
}
```

---

## 常见问题

### Q1: 提示"数据未下载"

**A:** 运行数据更新命令
```bash
rqalpha update_bundle
```

### Q2: 股票代码转换错误

**A:** 检查 `to_rqalpha_symbol()` 方法
- 上交所：`600000.XSHG`
- 深交所：`000001.XSHE`

### Q3: 预测文件格式不匹配

**A:** 确保预测文件包含以下列
```csv
instrument,score,date
000001,0.0012,2024-08-26
```

### Q4: 回测速度慢

**A:** 
- 减少回测时间范围
- 减少选股数量
- 使用 `matching_type: "current_bar"`

---

## 最佳实践

### 1. 数据准备

```bash
# 确保预测文件完整
ls data/predictions/ | wc -l

# 检查日期连续性
python -c "import pandas as pd; import os; files = sorted(os.listdir('data/predictions')); print(files[:5])"
```

### 2. 参数调优

```python
# 测试不同的选股数量
for top_n in [5, 10, 15, 20]:
    config['strategy']['top_n'] = top_n
    run_backtest()
```

### 3. 结果验证

```python
# 对比多个回测引擎的结果
results_rqalpha = run_rqalpha_backtest()
results_simple = run_simple_backtest()

# 确保收益率差异在合理范围内（<5%）
assert abs(results_rqalpha['returns'] - results_simple['returns']) < 0.05
```

---

## 扩展阅读

- [RQAlpha 官方文档](https://rqalpha.readthedocs.io/)
- [RQAlpha GitHub](https://github.com/ricequant/rqalpha)
- [量化交易入门](https://www.ricequant.com/welcome/tutorials)

---

## 示例代码

### 在 Jupyter Notebook 中使用

```python
# 加载 RQAlpha magic
%load_ext rqalpha

# 运行回测
%%rqalpha -s 2024-08-23 -e 2024-11-01 --account stock 10000000

from backtest.rqalpha_backtest import init, rebalance

# 查看结果
report['summary']
report['trades'].head(20)
```

### Python 脚本中使用

```python
from backtest.rqalpha_backtest import run_rqalpha_backtest

# 运行回测
report = run_rqalpha_backtest()

# 提取指标
total_returns = report['summary']['total_returns']
sharpe = report['summary']['sharpe']
max_drawdown = report['summary']['max_drawdown']

print(f"收益率: {total_returns:.2%}")
print(f"夏普: {sharpe:.2f}")
print(f"回撤: {max_drawdown:.2%}")
```

---

## 更新日志

### 2024-11-20
- ✅ 创建 RQAlpha 回测模块
- ✅ 集成模型预测结果
- ✅ 实现周频调仓策略
- ✅ 添加详细日志记录
- ✅ 输出完整交易明细

---

**提示**：首次使用前请确保已安装 RQAlpha 并下载数据！

```bash
pip install rqalpha
rqalpha update_bundle
python test_rqalpha.py
```






