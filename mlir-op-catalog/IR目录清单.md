# IR目录完整清单

## 🎯 项目概述

本项目成功生成了**43,006个IR文件**，包含MLIR和LLVM IR，为LLM4IR研究提供了丰富的实验数据。

## 📊 总体统计

| 类型 | 数量 | 说明 |
|------|------|------|
| **总MLIR文件** | 23,115 | 原始MLIR代码 |
| **总LLVM文件** | 19,891 | 编译后的LLVM IR |
| **总文件数** | 43,006 | 所有IR文件 |
| **单个操作** | 53 | 基础操作类型 |
| **成对操作** | 106 | 操作组合测试 |
| **总存储** | 约450MB | 包含所有文件 |

## 📁 详细目录结构

### 1. 单个操作 (out/single/)
- **MLIR文件**: 53个
- **操作类型**: 基础数学运算、激活函数、线性代数等
- **文件大小**: 472K
- **主要操作**:
  - 数学运算: add, sub, mul, div, pow, sqrt, exp, log
  - 激活函数: relu, sigmoid, tanh, gelu, swish, elu
  - 三角函数: sin, cos, asin, acos, atan, atan2
  - 比较操作: equal, greater, less, logical_and
  - 线性代数: matmul, conv2d_nhwc_hwcf
  - 池化操作: maxpool2d, avgpool2d

### 2. 单个操作LLVM (out/single_llvm/)
- **LLVM文件**: 53个
- **文件大小**: 1.6M
- **说明**: 单个操作编译后的LLVM IR

### 3. 完整单个操作 (out/single_complete/)
- **MLIR文件**: 142个
- **文件大小**: 1.2M
- **说明**: 扩展的单个操作集合

### 4. 完整单个操作LLVM (out/single_complete_llvm/)
- **LLVM文件**: 142个
- **文件大小**: 1.7M
- **说明**: 完整单个操作的LLVM IR

### 5. 成对操作 (out/pairs/)
- **MLIR文件**: 2,756个
- **文件大小**: 22M
- **说明**: 成对操作的MLIR代码

### 6. 成对操作LLVM (out/pairs_llvm/)
- **LLVM文件**: 96个
- **文件大小**: 2.8M
- **说明**: 成对操作的LLVM IR

### 7. 完整成对操作 (out/pairs_complete/)
- **MLIR文件**: 20,164个
- **文件大小**: 159M
- **说明**: 扩展的成对操作集合

### 8. 完整成对操作LLVM (out/pairs_complete_llvm/)
- **LLVM文件**: 19,600个
- **文件大小**: 234M
- **说明**: 完整成对操作的LLVM IR

## 🔧 操作类型分类

### 基础数学运算
- **算术运算**: add, sub, mul, div, mod
- **幂运算**: pow, sqrt, rsqrt, cbrt
- **指数对数**: exp, log, log2, log10
- **三角函数**: sin, cos, tan, asin, acos, atan, atan2
- **双曲函数**: sinh, cosh, tanh, asinh, acosh, atanh

### 激活函数
- **基础激活**: relu, sigmoid, tanh
- **高级激活**: gelu, swish, elu, leaky_relu
- **硬激活**: hard_sigmoid, hard_tanh
- **其他激活**: mish, softplus, softsign

### 比较和逻辑运算
- **比较运算**: equal, not_equal, greater, greater_equal, less, less_equal
- **逻辑运算**: logical_and, logical_or, logical_xor, logical_not

### 线性代数
- **矩阵运算**: matmul
- **卷积运算**: conv2d_nhwc_hwcf
- **池化运算**: maxpool2d, avgpool2d
- **归约运算**: reduce_sum

### 成对操作组合
- **激活函数组合**: relu_gelu, leaky_relu_sigmoid等
- **数学运算组合**: add_relu, mul_tanh等
- **复杂组合**: 测试操作间的兼容性

## 📐 数据格式规范

### 输入输出形状
- **4D张量**: `tensor<1x8x8x8xf32>` (主要格式)
- **2D张量**: `tensor<16x16xf32>` (矩阵运算)
- **卷积核**: `tensor<3x3x8x8xf32>` (卷积操作)
- **输出张量**: `tensor<1x6x6x8xf32>`, `tensor<1x4x4x8xf32>`等

### MLIR方言
- **主要方言**: linalg (线性代数)
- **支持方言**: arith, math, tensor, memref
- **操作类型**: linalg.generic, linalg.matmul, linalg.conv_2d_nhwc_hwcf

### LLVM IR格式
- **标准格式**: 标准LLVM IR
- **函数签名**: 统一的tensor结构
- **内存管理**: 自动内存分配和释放

## 🧪 实验建议

### 按复杂度分组
1. **简单操作组** (1x8x8x8xf32)
   - 元素级运算: add, sub, mul, div
   - 激活函数: relu, sigmoid, tanh
   - 数学函数: exp, log, sqrt

2. **中等操作组** (16x16xf32)
   - 矩阵乘法: matmul
   - 线性代数运算

3. **复杂操作组** (多形状)
   - 卷积: conv2d_nhwc_hwcf
   - 池化: maxpool2d, avgpool2d
   - 成对操作组合

### 实验流程
1. **基础验证**: 使用单个操作验证MLIR→LLVM转换
2. **性能测试**: 测试不同操作的执行效率
3. **组合测试**: 验证成对操作的兼容性
4. **优化研究**: 分析不同优化传递的效果

## 📋 文件命名规范

### 单个操作
- **MLIR**: `{operation}_{shape}.mlir`
- **LLVM**: `{operation}_{shape}.ll`
- **示例**: `add_N1_H8_W8_C8.mlir`

### 成对操作
- **MLIR**: `{op1}_{shape}_then_{op2}_{shape}.mlir`
- **LLVM**: `{op1}_{shape}_then_{op2}_{shape}.ll`
- **示例**: `relu_N1_H8_W8_C8_then_gelu_N1_H8_W8_C8.ll`

## 🚀 使用工具

### 便捷脚本
- `./run_mlir.sh list` - 列出所有操作
- `./run_mlir.sh run <op>` - 运行特定操作
- `./run_mlir.sh stats` - 查看统计信息
- `./run_mlir.sh demo` - 运行完整演示

### 分析工具
- `extract_io_info.py` - 提取输入输出信息
- `analyze_ir_catalog.py` - 分析目录结构
- `ir_io_info.json` - 详细的IO信息
- `ir_catalog_analysis.json` - 目录分析数据

## 📈 项目价值

1. **丰富的实验数据**: 43,006个IR文件覆盖深度学习主要操作
2. **标准化的格式**: 统一的MLIR和LLVM IR格式
3. **完整的工具链**: 从生成到运行的完整工具
4. **详细的文档**: 完整的操作清单和使用说明
5. **可扩展性**: 易于添加新的操作和测试用例

这个IR目录为LLM4IR研究提供了坚实的基础，可以支持各种实验和分析需求。
