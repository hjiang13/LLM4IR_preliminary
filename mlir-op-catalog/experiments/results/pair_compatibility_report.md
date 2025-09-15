# 成对操作兼容性测试报告

**测试时间**: 2025-09-14T23:15:28.345321
**测试类型**: pair_compatibility
**测试对数**: 15
**兼容对数**: 14
**不兼容对数**: 1
**兼容率**: 93.3%

## 📊 操作兼容性矩阵

| 操作 | 总配对数 | 兼容配对数 | 兼容率 | 平均分数 |
|------|----------|------------|--------|----------|
| add_N1_H8_W8_C8 | 1 | 0 | 0.0% | 0.000 |
| atan2 | 1 | 1 | 100.0% | 1.000 |
| equal | 1 | 1 | 100.0% | 1.000 |
| gelu | 1 | 1 | 100.0% | 1.000 |
| leaky | 10 | 10 | 100.0% | 1.000 |
| mul_N1_H8_W8_C8 | 1 | 0 | 0.0% | 0.000 |
| relu | 4 | 4 | 100.0% | 1.000 |
| relu_atanh | 1 | 1 | 100.0% | 1.000 |
| relu_cbrt | 1 | 1 | 100.0% | 1.000 |
| relu_div | 1 | 1 | 100.0% | 1.000 |
| relu_elu | 1 | 1 | 100.0% | 1.000 |
| relu_exp | 1 | 1 | 100.0% | 1.000 |
| relu_gelu | 1 | 1 | 100.0% | 1.000 |
| relu_log10 | 1 | 1 | 100.0% | 1.000 |
| relu_logical_or | 1 | 1 | 100.0% | 1.000 |
| relu_relu | 1 | 1 | 100.0% | 1.000 |
| relu_sin | 1 | 1 | 100.0% | 1.000 |
| sqrt | 1 | 1 | 100.0% | 1.000 |

## 🔗 详细配对结果

| 配对 | 操作1 | 操作2 | 兼容性分数 | 状态 |
|------|-------|-------|------------|------|
| relu_sqrt | relu | sqrt | 1.000 | ✅ |
| leaky_relu_gelu | leaky | relu_gelu | 1.000 | ✅ |
| leaky_relu_exp | leaky | relu_exp | 1.000 | ✅ |
| leaky_relu_elu | leaky | relu_elu | 1.000 | ✅ |
| leaky_relu_log10 | leaky | relu_log10 | 1.000 | ✅ |
| relu_atan2 | relu | atan2 | 1.000 | ✅ |
| leaky_relu_sin | leaky | relu_sin | 1.000 | ✅ |
| leaky_relu_relu | leaky | relu_relu | 1.000 | ✅ |
| leaky_relu_logical_or | leaky | relu_logical_or | 1.000 | ✅ |
| leaky_relu_atanh | leaky | relu_atanh | 1.000 | ✅ |
| relu_gelu | relu | gelu | 1.000 | ✅ |
| add_N1_H8_W8_C8_then_mul_N1_H8_W8_C8 | add_N1_H8_W8_C8 | mul_N1_H8_W8_C8 | 0.000 | ❌ |
| relu_equal | relu | equal | 1.000 | ✅ |
| leaky_relu_cbrt | leaky | relu_cbrt | 1.000 | ✅ |
| leaky_relu_div | leaky | relu_div | 1.000 | ✅ |

## 💡 推荐建议

### 高兼容性操作

- **relu**: 兼容率 100.0%
- **sqrt**: 兼容率 100.0%
- **leaky**: 兼容率 100.0%
- **relu_gelu**: 兼容率 100.0%
- **relu_exp**: 兼容率 100.0%
- **relu_elu**: 兼容率 100.0%
- **relu_log10**: 兼容率 100.0%
- **atan2**: 兼容率 100.0%
- **relu_sin**: 兼容率 100.0%
- **relu_relu**: 兼容率 100.0%
- **relu_logical_or**: 兼容率 100.0%
- **relu_atanh**: 兼容率 100.0%
- **gelu**: 兼容率 100.0%
- **equal**: 兼容率 100.0%
- **relu_cbrt**: 兼容率 100.0%
- **relu_div**: 兼容率 100.0%

### 问题操作

- **add_N1_H8_W8_C8**: 兼容率 0.0%
- **mul_N1_H8_W8_C8**: 兼容率 0.0%

### 最佳配对

- **relu_sqrt**: relu + sqrt
- **leaky_relu_gelu**: leaky + relu_gelu
- **leaky_relu_exp**: leaky + relu_exp
- **leaky_relu_elu**: leaky + relu_elu
- **leaky_relu_log10**: leaky + relu_log10
- **relu_atan2**: relu + atan2
- **leaky_relu_sin**: leaky + relu_sin
- **leaky_relu_relu**: leaky + relu_relu
- **leaky_relu_logical_or**: leaky + relu_logical_or
- **leaky_relu_atanh**: leaky + relu_atanh

### 应避免的配对

- **add_N1_H8_W8_C8_then_mul_N1_H8_W8_C8**: add_N1_H8_W8_C8 + mul_N1_H8_W8_C8 (分数: 0.000)

## ❌ 错误分析

| 错误类型 | 出现次数 |
|----------|----------|
