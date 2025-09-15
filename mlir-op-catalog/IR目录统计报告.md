# IR目录统计报告

## 📊 总体统计

- **总MLIR文件**: 0
- **总LLVM文件**: 19,891
- **总文件数**: 19,891
- **单个操作数**: 53
- **成对操作数**: 106

## 📁 目录统计

| 目录 | MLIR文件 | LLVM文件 | 总文件数 |
|------|----------|----------|----------|
| single | 53 | 5 | 58 |
| single_llvm | 53 | 53 | 106 |
| single_complete | 142 | 0 | 142 |
| single_complete_llvm | 0 | 142 | 142 |
| pairs | 2,756 | 0 | 2,756 |
| pairs_llvm | 100 | 96 | 196 |
| pairs_complete | 20,164 | 0 | 20,164 |
| pairs_complete_llvm | 0 | 19,600 | 19,600 |

## 🔧 操作类型

### 单个操作 (前20个)

1. **exp**
2. **mish**
3. **softsign**
4. **add**
5. **log2**
6. **greater**
7. **asin**
8. **logical_xor**
9. **cosh**
10. **sinh**
11. **log**
12. **greater_equal**
13. **equal**
14. **sigmoid**
15. **conv2d_nhwc_hwcf**
16. **atan2**
17. **asinh**
18. **cos**
19. **cbrt**
20. **matmul**

... 还有 33 个操作

### 成对操作 (前20个)

1. **leaky_relu_greater_equal**
2. **leaky_relu_tanh**
3. **relu_softplus**
4. **relu_gelu**
5. **relu_logical_and**
6. **leaky_relu_mish**
7. **relu_less_equal**
8. **leaky_relu_greater**
9. **relu_equal**
10. **leaky_relu_asinh**
11. **leaky_relu_equal**
12. **leaky_relu_atan2**
13. **relu_asin**
14. **leaky_relu_softplus**
15. **leaky_relu_log2**
16. **leaky_relu_logical_xor**
17. **relu_leaky_relu**
18. **leaky_relu_less_equal**
19. **leaky_relu_sigmoid**
20. **relu_log10**

... 还有 86 个操作

## 💾 文件大小统计

```bash
# 各目录大小
du -sh out/single
du -sh out/single_llvm
du -sh out/single_complete
du -sh out/single_complete_llvm
du -sh out/pairs
du -sh out/pairs_llvm
du -sh out/pairs_complete
du -sh out/pairs_complete_llvm
```

## 🧪 实验建议

### 按复杂度分类

1. **简单操作** (1x8x8x8xf32):
   - 元素级运算: add, sub, mul, div
   - 激活函数: relu, sigmoid, tanh
   - 数学函数: exp, log, sqrt

2. **中等操作** (16x16xf32):
   - 矩阵乘法: matmul
   - 线性代数运算

3. **复杂操作** (多形状):
   - 卷积: conv2d_nhwc_hwcf
   - 池化: maxpool2d, avgpool2d
   - 成对操作组合

### 实验分组

1. **基础运算组**: 数学运算 + 激活函数
2. **线性代数组**: 矩阵运算 + 卷积
3. **组合运算组**: 成对操作测试
4. **完整测试组**: 所有操作的综合测试

### 数据格式

所有IR文件都使用标准格式：
- **MLIR**: 使用linalg方言
- **LLVM IR**: 标准LLVM IR格式
- **输入输出**: 统一的tensor格式

