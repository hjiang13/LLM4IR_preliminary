# MLIR完整操作目录编译结果报告

**生成时间**: 2025-09-02 00:29:02

## 📊 编译摘要

- **总操作数**: 142
- **成功编译**: 142
- **编译成功率**: 100.0%

## 🔧 编译配置

- **输入目录**: out/single_complete
- **输出目录**: out/single_complete_llvm
- **编译管道**: config/pipeline_linalg_to_llvm.txt

## 📈 按方言统计

### ARITH 方言

- **操作总数**: 28
- **编译成功**: 28
- **成功率**: 100.0%

**包含操作**:
- `arith.constant`
- `arith.addi`
- `arith.subi`
- `arith.muli`
- `arith.divui`
- `arith.divsi`
- `arith.remui`
- `arith.remsi`
- `arith.addf`
- `arith.subf`
- `arith.mulf`
- `arith.divf`
- `arith.maxf`
- `arith.minf`
- `arith.maxsi`
- `arith.minsi`
- `arith.andi`
- `arith.ori`
- `arith.xori`
- `arith.shli`
- `arith.shrsi`
- `arith.shrui`
- `arith.cmpi`
- `arith.cmpf`
- `arith.bitcast`
- `arith.index_cast`
- `arith.fptosi`
- `arith.sitofp`

### MATH 方言

- **操作总数**: 14
- **编译成功**: 14
- **成功率**: 100.0%

**包含操作**:
- `math.absf`
- `math.copysign`
- `math.ceil`
- `math.floor`
- `math.round`
- `math.roundeven`
- `math.sqrt`
- `math.rsqrt`
- `math.log`
- `math.log1p`
- `math.exp`
- `math.exp2`
- `math.powf`
- `math.tanh`

### MEMREF 方言

- **操作总数**: 14
- **编译成功**: 14
- **成功率**: 100.0%

**包含操作**:
- `memref.alloc`
- `memref.alloca`
- `memref.dealloc`
- `memref.load`
- `memref.store`
- `memref.subview`
- `memref.cast`
- `memref.reinterpret_cast`
- `memref.view`
- `memref.reshape`
- `memref.expand_shape`
- `memref.collapse_shape`
- `memref.copy`
- `memref.global`

### TENSOR 方言

- **操作总数**: 12
- **编译成功**: 12
- **成功率**: 100.0%

**包含操作**:
- `tensor.extract`
- `tensor.insert`
- `tensor.extract_slice`
- `tensor.insert_slice`
- `tensor.cast`
- `tensor.dim`
- `tensor.generate`
- `tensor.empty`
- `tensor.from_elements`
- `tensor.reshape`
- `tensor.expand_shape`
- `tensor.collapse_shape`

### LINALG 方言

- **操作总数**: 16
- **编译成功**: 16
- **成功率**: 100.0%

**包含操作**:
- `linalg.generic`
- `linalg.fill`
- `linalg.copy`
- `linalg.transpose`
- `linalg.matmul`
- `linalg.matvec`
- `linalg.batch_matmul`
- `linalg.conv_1d_nwc_wcf`
- `linalg.conv_2d_nchw_fchw`
- `linalg.conv_2d_nhwc_hwcf`
- `linalg.conv_3d_ndhwc_dhwcf`
- `linalg.depthwise_conv_2d_nhwc_hwc`
- `linalg.pooling_nchw_max`
- `linalg.pooling_nhwc_max`
- `linalg.pooling_nchw_sum`
- `linalg.pooling_nhwc_sum`

### SCF 方言

- **操作总数**: 10
- **编译成功**: 10
- **成功率**: 100.0%

**包含操作**:
- `scf.for`
- `scf.while`
- `scf.if`
- `scf.yield`
- `scf.parallel`
- `scf.forall`
- `scf.foreach_thread`
- `scf.in_parallel`
- `scf.execute_region`
- `scf.index_switch`

### AFFINE 方言

- **操作总数**: 10
- **编译成功**: 10
- **成功率**: 100.0%

**包含操作**:
- `affine.for`
- `affine.if`
- `affine.parallel`
- `affine.yield`
- `affine.apply`
- `affine.min`
- `affine.max`
- `affine.load`
- `affine.store`
- `affine.vector_load`

### VECTOR 方言

- **操作总数**: 14
- **编译成功**: 14
- **成功率**: 100.0%

**包含操作**:
- `vector.broadcast`
- `vector.splat`
- `vector.extract`
- `vector.insert`
- `vector.extract_strided_slice`
- `vector.insert_strided_slice`
- `vector.shape_cast`
- `vector.bitcast`
- `vector.transfer_read`
- `vector.transfer_write`
- `vector.gather`
- `vector.scatter`
- `vector.reduction`
- `vector.contract`

### CF 方言

- **操作总数**: 5
- **编译成功**: 5
- **成功率**: 100.0%

**包含操作**:
- `cf.br`
- `cf.cond_br`
- `cf.switch`
- `cf.assert`
- `cf.unreachable`

### FUNC 方言

- **操作总数**: 3
- **编译成功**: 3
- **成功率**: 100.0%

**包含操作**:
- `func.func`
- `func.call`
- `func.return`

### TOSA 方言

- **操作总数**: 8
- **编译成功**: 8
- **成功率**: 100.0%

**包含操作**:
- `tosa.add`
- `tosa.mul`
- `tosa.relu`
- `tosa.clamp`
- `tosa.conv2d`
- `tosa.depthwise_conv2d`
- `tosa.max_pool2d`
- `tosa.reshape`

### STABLEHLO 方言

- **操作总数**: 8
- **编译成功**: 8
- **成功率**: 100.0%

**包含操作**:
- `stablehlo.add`
- `stablehlo.multiply`
- `stablehlo.maximum`
- `stablehlo.compare`
- `stablehlo.reshape`
- `stablehlo.transpose`
- `stablehlo.broadcast_in_dim`
- `stablehlo.convolution`

