# 基线测试报告

**测试时间**: 2025-09-14T23:12:05.421648
**测试类型**: baseline_validation

## 📊 总体统计

- **总测试数**: 63
- **通过测试**: 62
- **失败测试**: 1
- **成功率**: 98.4%

## 🔧 单个操作测试结果

| 操作 | MLIR有效 | LLVM有效 | 优化有效 | 状态 |
|------|----------|----------|----------|------|
| exp | ✅ | ✅ | ✅ | ✅ |
| mish | ✅ | ✅ | ✅ | ✅ |
| softsign | ✅ | ✅ | ✅ | ✅ |
| add | ✅ | ✅ | ✅ | ✅ |
| log2 | ✅ | ✅ | ✅ | ✅ |
| greater | ✅ | ✅ | ✅ | ✅ |
| asin | ✅ | ✅ | ✅ | ✅ |
| logical_xor | ✅ | ✅ | ✅ | ✅ |
| cosh | ✅ | ✅ | ✅ | ✅ |
| sinh | ✅ | ✅ | ✅ | ✅ |
| log | ✅ | ✅ | ✅ | ✅ |
| greater_equal | ✅ | ✅ | ✅ | ✅ |
| equal | ✅ | ✅ | ✅ | ✅ |
| sigmoid | ✅ | ✅ | ✅ | ✅ |
| conv2d_nhwc_hwcf | ✅ | ✅ | ✅ | ✅ |
| atan2 | ✅ | ✅ | ✅ | ✅ |
| asinh | ✅ | ✅ | ✅ | ✅ |
| cos | ✅ | ✅ | ✅ | ✅ |
| cbrt | ✅ | ✅ | ✅ | ✅ |
| matmul | ❌ | ❌ | ❌ | ❌ |
| sqrt | ✅ | ✅ | ✅ | ✅ |
| atan | ✅ | ✅ | ✅ | ✅ |
| clamp | ✅ | ✅ | ✅ | ✅ |
| reduce_sum | ✅ | ✅ | ✅ | ✅ |
| logical_not | ✅ | ✅ | ✅ | ✅ |
| mul | ✅ | ✅ | ✅ | ✅ |
| avgpool2d | ✅ | ✅ | ✅ | ✅ |
| acosh | ✅ | ✅ | ✅ | ✅ |
| less | ✅ | ✅ | ✅ | ✅ |
| tanh | ✅ | ✅ | ✅ | ✅ |
| less_equal | ✅ | ✅ | ✅ | ✅ |
| hard_sigmoid | ✅ | ✅ | ✅ | ✅ |
| atanh | ✅ | ✅ | ✅ | ✅ |
| mod | ✅ | ✅ | ✅ | ✅ |
| gelu | ✅ | ✅ | ✅ | ✅ |
| rsqrt | ✅ | ✅ | ✅ | ✅ |
| softplus | ✅ | ✅ | ✅ | ✅ |
| tan | ✅ | ✅ | ✅ | ✅ |
| log10 | ✅ | ✅ | ✅ | ✅ |
| swish | ✅ | ✅ | ✅ | ✅ |
| acos | ✅ | ✅ | ✅ | ✅ |
| div | ✅ | ✅ | ✅ | ✅ |
| sub | ✅ | ✅ | ✅ | ✅ |
| pow | ✅ | ✅ | ✅ | ✅ |
| hard_tanh | ✅ | ✅ | ✅ | ✅ |
| not_equal | ✅ | ✅ | ✅ | ✅ |
| sin | ✅ | ✅ | ✅ | ✅ |
| relu | ✅ | ✅ | ✅ | ✅ |
| leaky_relu | ✅ | ✅ | ✅ | ✅ |
| logical_and | ✅ | ✅ | ✅ | ✅ |
| elu | ✅ | ✅ | ✅ | ✅ |
| logical_or | ✅ | ✅ | ✅ | ✅ |
| maxpool2d | ✅ | ✅ | ✅ | ✅ |

## 🔗 成对操作测试结果

| 操作 | 有效文件 | 总文件 | 成功率 |
|------|----------|--------|--------|
| relu_less | 1 | 1 | 100.0% |
| leaky_relu_mish | 1 | 1 | 100.0% |
| leaky_relu_acos | 1 | 1 | 100.0% |
| leaky_relu_greater_equal | 1 | 1 | 100.0% |
| leaky_relu_acosh | 1 | 1 | 100.0% |
| relu_clamp | 1 | 1 | 100.0% |
| relu_atanh | 1 | 1 | 100.0% |
| leaky_relu_not_equal | 1 | 1 | 100.0% |
| relu_gelu | 1 | 1 | 100.0% |
| leaky_relu_greater | 1 | 1 | 100.0% |

## ❌ 错误信息

- MLIR文件不存在
- LLVM文件不存在
