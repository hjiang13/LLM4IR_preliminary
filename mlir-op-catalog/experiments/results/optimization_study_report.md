# 优化研究报告

**研究时间**: 2025-09-14T23:14:18.204169
**研究类型**: optimization_effectiveness
**分析操作数**: 12

## 🔧 传递组合效果

| 组合 | 平均时间(ms) | 成功率 | 输出大小 | 排名 |
|------|-------------|--------|----------|------|
| minimal | 19.50 | 100.0% | 14 | 1 |
| basic | 20.13 | 100.0% | 14 | 2 |
| intermediate | 21.80 | 100.0% | 14 | 3 |
| advanced | 22.22 | 100.0% | 14 | 4 |
| full | 24.48 | 100.0% | 73 | 5 |

## 💡 优化建议

### 总体建议

- **最佳综合**: minimal
- **最快速度**: minimal
- **最高可靠性**: minimal

### 使用场景建议

- **development**: basic
- **production**: advanced
- **maximum_optimization**: full
- **quick_testing**: minimal

### 操作特定建议

| 操作 | 推荐传递组合 |
|------|-------------|
| conv2d_nhwc_hwcf | minimal |
| sqrt | minimal |
| acosh | minimal |
| atan | basic |
| exp | basic |
| cosh | minimal |
| sinh | minimal |
| mod | minimal |
| elu | minimal |
| not_equal | minimal |
| atanh | minimal |
| log2 | minimal |

## 📊 详细分析

### 时间效率排名

1. **minimal**
2. **basic**
3. **intermediate**
4. **advanced**
5. **full**

### 操作优化效果

| 操作 | 最佳传递 | 时间改进 | 输出变化 |
|------|----------|----------|----------|
| conv2d_nhwc_hwcf | minimal | -27.7% | +112 |
| sqrt | minimal | -21.9% | +55 |
| acosh | minimal | -18.6% | +55 |
| atan | basic | -19.1% | +55 |
| exp | basic | -25.9% | +55 |
| cosh | minimal | -13.3% | +55 |
| sinh | minimal | -21.5% | +55 |
| mod | minimal | -25.3% | +55 |
| elu | minimal | -23.5% | +55 |
| not_equal | minimal | -34.1% | +55 |
| atanh | minimal | -42.9% | +55 |
| log2 | minimal | -33.3% | +55 |
