# IR输入输出信息统计报告

## 📊 总体统计

- **总MLIR文件**: 23115
- **总LLVM文件**: 19891
- **总文件数**: 43006

## 🔧 MLIR文件统计

### 操作类型分布

- **generic**: 23113 次
- **yield**: 23113 次
- **addf**: 957 次
- **mulf**: 771 次
- **exp**: 770 次
- **subf**: 669 次
- **sqrt**: 668 次
- **cmpf**: 564 次
- **divf**: 490 次
- **constant**: 207 次
- **empty**: 195 次
- **maximumf**: 105 次
- **cos**: 104 次
- **log**: 104 次
- **sin**: 104 次
- **tan**: 104 次
- **tanh**: 104 次
- **conv_2d_nhwc_hwcf**: 1 次
- **matmul**: 1 次

### 输入输出类型统计

#### 输入类型

- **tensor<1x8x8x8xf32>**: 23118 次
- **tensor<16x16xf32>**: 2 次
- **tensor<3x3x8x8xf32>**: 1 次

#### 输出类型

- **tensor<1x8x8x8xf32>**: 23110 次
- **tensor<1x4x4x8xf32>**: 2 次
- **tensor<1x6x6x8xf32>**: 1 次
- **tensor<16x16xf32>**: 1 次
- **tensor<1x8xf32>**: 1 次

## 📝 示例文件

### MLIR示例

**文件**: `out/single/acos/acos_N1_H8_W8_C8.mlir`
- 输入: 1 个
- 输出: 1 个
- 操作: generic, yield, addf, empty

**文件**: `out/single/acosh/acosh_N1_H8_W8_C8.mlir`
- 输入: 1 个
- 输出: 1 个
- 操作: generic, yield, addf, empty

**文件**: `out/single/add/add_N1_H8_W8_C8.mlir`
- 输入: 2 个
- 输出: 1 个
- 操作: generic, yield, addf, empty

**文件**: `out/single/asin/asin_N1_H8_W8_C8.mlir`
- 输入: 1 个
- 输出: 1 个
- 操作: generic, yield, addf, empty

**文件**: `out/single/asinh/asinh_N1_H8_W8_C8.mlir`
- 输入: 1 个
- 输出: 1 个
- 操作: generic, yield, addf, empty

### LLVM示例

**文件**: `out/single_llvm/acos/acos_N1_H8_W8_C8/acos_N1_H8_W8_C8.ll`
- 函数: 0 个
- 输入: 0 个

**文件**: `out/single_llvm/acosh/acosh_N1_H8_W8_C8/acosh_N1_H8_W8_C8.ll`
- 函数: 0 个
- 输入: 0 个

**文件**: `out/single_llvm/add/add_N1_H8_W8_C8/add_N1_H8_W8_C8.ll`
- 函数: 0 个
- 输入: 0 个

**文件**: `out/single_llvm/asin/asin_N1_H8_W8_C8/asin_N1_H8_W8_C8.ll`
- 函数: 0 个
- 输入: 0 个

**文件**: `out/single_llvm/asinh/asinh_N1_H8_W8_C8/asinh_N1_H8_W8_C8.ll`
- 函数: 0 个
- 输入: 0 个

