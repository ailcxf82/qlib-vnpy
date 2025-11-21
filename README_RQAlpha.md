# RQAlpha 回测快速入门

## 🚀 三步开始使用

### 1️⃣ 安装 RQAlpha

```bash
pip install rqalpha
rqalpha update_bundle  # 下载数据（首次使用，约需10分钟）
```

### 2️⃣ 运行回测

```bash
cd D:\lianghuatouzi\Qlib1114
python test_rqalpha.py
```

### 3️⃣ 查看结果

```bash
# 回测摘要
cat data/backtest_results/rqalpha_summary.csv

# 交易明细
cat data/backtest_results/rqalpha_trades.csv

# 权益曲线
cat data/backtest_results/rqalpha_portfolio.csv
```

---

## 📊 预期输出

### 控制台输出

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

交易明细（前20条）
================================================================================
日期                 股票代码      方向  数量    价格     手续费
2024-08-26 15:00    000001.XSHE  买入  12400   15.23    189.45
2024-08-26 15:00    600000.XSHG  买入  10000   23.47    234.67
...
```

---

## 🎯 核心特性

### ✅ 基于模型预测选股
- 自动读取 `data/predictions/pred_YYYY-MM-DD.csv`
- 按分数排序，选择 Top N 只股票

### ✅ 周频调仓
- 每周一（Monday）自动调仓
- 等权重配置目标股票

### ✅ 完整交易记录
- 每笔交易的详细信息
- 持仓变化跟踪
- 盈亏统计

### ✅ 专业回测指标
- 30+ 专业指标
- 与基准对比
- 风险调整收益

---

## 📁 文件结构

```
backtest/
├── rqalpha_backtest.py      # RQAlpha 回测主程序
├── run_vnpy_backtest.py     # 原 SimpleBacktester（备选）
└── vnpy_backtest_engine.py  # vn.py 引擎（实验性）

data/
├── predictions/              # 模型预测结果
│   ├── pred_2024-08-23.csv
│   └── ...
└── backtest_results/         # 回测结果
    ├── rqalpha_portfolio.csv    # 权益曲线
    ├── rqalpha_trades.csv       # 交易明细
    ├── rqalpha_positions.csv    # 持仓记录
    └── rqalpha_summary.csv      # 回测摘要

docs/
└── RQAlpha回测使用说明.md   # 详细文档

test_rqalpha.py              # 快速测试脚本
```

---

## ⚙️ 配置参数

编辑 `config/backtest.yaml`：

```yaml
backtest:
  start_date: "2024-08-23"
  end_date: "2024-11-01"
  capital: 10000000        # 初始资金

strategy:
  top_n: 10                # 选股数量

costs:
  commission: 0.0003       # 佣金 0.03%
  slippage: 0.0002         # 滑点 0.02%
```

---

## 🔧 常见问题

### Q: 提示"rqalpha: command not found"
**A:** RQAlpha 未正确安装
```bash
pip install rqalpha
which rqalpha  # 检查是否安装成功
```

### Q: 提示"数据未下载"
**A:** 需要下载历史数据
```bash
rqalpha update_bundle
```

### Q: 预测文件不存在
**A:** 先运行预测
```bash
python pipeline/run_predict.py
```

### Q: 股票代码格式错误
**A:** RQAlpha 使用的格式：
- 上交所：`600000.XSHG`
- 深交所：`000001.XSHE`

---

## 📚 更多信息

- 📖 [详细使用文档](docs/RQAlpha回测使用说明.md)
- 🌐 [RQAlpha 官方文档](https://rqalpha.readthedocs.io/)
- 💻 [GitHub 仓库](https://github.com/ricequant/rqalpha)

---

## 🎓 学习资源

### RQAlpha 教程
- [10分钟教程](https://rqalpha.readthedocs.io/zh-cn/latest/intro/tutorial.html)
- [API 文档](https://rqalpha.readthedocs.io/zh-cn/latest/api/base_api.html)
- [策略示例](https://rqalpha.readthedocs.io/zh-cn/latest/intro/examples.html)

### Jupyter Notebook 使用
参考：https://rqalpha.readthedocs.io/zh-cn/latest/notebooks/run-rqalpha-in-ipython.html

```python
# 在 Jupyter 中使用
%load_ext rqalpha

%%rqalpha -s 2024-08-23 -e 2024-11-01 --account stock 10000000
# 策略代码...
```

---

## ✨ 优势对比

| 特性 | RQAlpha | SimpleBacktester | vn.py |
|------|---------|------------------|-------|
| 专业性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 数据完整性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 易用性 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 社区支持 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| 文档质量 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

**推荐使用 RQAlpha 进行专业的量化回测！**

---

**现在就开始使用吧！**

```bash
python test_rqalpha.py
```




