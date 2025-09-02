# MLIR高级操作目录 → LLVM IR

这个项目旨在生成和编译≥120个高级MLIR操作到LLVM IR，并测试所有成对组合的兼容性。

## 🚀 快速开始

### 1. 环境检查

首先检查您的环境是否满足要求：

```bash
bash env/check_env.sh
```

这将验证以下工具是否可用：
- `mlir-opt` - MLIR优化工具
- `mlir-translate` - MLIR转换工具  
- `llvm-as` - LLVM汇编器
- `opt` - LLVM优化器
- `python3` ≥ 3.9
- Python包：`jinja2`, `pyyaml`, `pandas`

### 2. 生成操作实例

```bash
python3 scripts/enumerate_ops.py
```

这将从配置文件生成具体的操作实例，并保存到 `out/cases_single.json` 和 `out/cases_pairs.json`。

### 3. 生成单个操作MLIR

```bash
python3 scripts/gen_singletons.py
```

这将为每个操作实例生成对应的MLIR文件，保存到 `out/single/<op_id>/` 目录。

### 4. 编译单个操作

```bash
# 编译单个文件
python3 scripts/run_pass_pipeline.py --input out/single/<op>/<case>.mlir --out-ll out/single/<op>/<case>.ll --pipeline linalg

# 批量编译所有单个操作
find out/single -name '*.mlir' -print0 | xargs -0 -I{} python3 scripts/run_pass_pipeline.py --input {} --out-ll {}.ll --pipeline linalg || true
```

### 5. 生成成对操作

```bash
python3 scripts/gen_pairs.py
```

这将生成成对操作的MLIR文件，保存到 `out/pairs/` 目录。

### 6. 编译成对操作

```bash
# 批量编译所有成对操作
find out/pairs -name '*.mlir' -print0 | xargs -0 -I{} python3 scripts/run_pass_pipeline.py --input {} --out-ll {}.ll --pipeline linalg || true
```

### 7. 生成结果汇总

```bash
python3 scripts/summarize_results.py
```

这将生成编译结果的汇总报告，包括：
- `out/reports/summary_single.md` - 单个操作结果
- `out/reports/summary_pairs.md` - 成对操作结果
- `out/reports/results_single.csv` - 单个操作详细结果
- `out/reports/results_pairs.csv` - 成对操作详细结果

## 📁 项目结构

```
mlir-op-catalog/
├── README.md                    # 项目说明
├── env/
│   └── check_env.sh            # 环境检查脚本
├── config/
│   ├── ops.yaml                # 操作目录配置 (≥120个操作)
│   ├── dtypes.yaml             # 数据类型配置
│   ├── shapes.yaml             # 形状参数配置
│   ├── pipeline_linalg_to_llvm.txt    # Linalg→LLVM传递管道
│   └── pipeline_tf_to_llvm.txt        # TF→LLVM传递管道
├── templates/
│   ├── common/                 # 通用模板
│   └── linalg/                 # Linalg操作模板
├── scripts/                    # Python脚本
├── out/                        # 输出目录
│   ├── single/                 # 单个操作结果
│   ├── pairs/                  # 成对操作结果
│   ├── logs/                   # 编译日志
│   └── reports/                # 汇总报告
└── data/                       # 数据文件
```

## 🔧 传递管道

### Linalg → LLVM IR

使用 `config/pipeline_linalg_to_llvm.txt` 中的传递管道：

```
--one-shot-bufferize=bufferize-function-boundaries
--arith-bufferize
--tensor-bufferize
--finalizing-bufferize
--convert-linalg-to-loops
--lower-affine
--convert-scf-to-cf
--convert-math-to-llvm
--convert-arith-to-llvm
--convert-func-to-llvm
--memref-expand
--convert-memref-to-llvm
--reconcile-unrealized-casts
```

### TF → LLVM IR (可选)

如果 `tf-mlir-translate` 工具可用，可以使用 `config/pipeline_tf_to_llvm.txt` 中的传递管道。

## 📊 操作类型

项目包含以下操作类型：

### 元素级操作
- **激活函数**: `relu`, `tanh`, `sigmoid`, `silu`, `gelu`
- **算术运算**: `add`, `sub`, `mul`, `div`, `pow`
- **数学函数**: `exp`, `log`, `sqrt`, `abs`
- **其他**: `clamp`, `round`, `floor`, `ceil`

### 线性代数
- **矩阵乘法**: `matmul`, `batch_matmul`
- **卷积**: `conv2d_nhwc_hwcf`, `conv3d_ndhwc_dhwcf`
- **池化**: `maxpool2d`, `avgpool2d`, `global_pooling`

### 归约操作
- **归约**: `reduce_sum`, `reduce_max`, `reduce_min`, `reduce_mean`
- **归一化**: `softmax`, `layer_norm`, `batch_norm`

### 张量操作
- **形状变换**: `reshape`, `transpose`, `expand_dims`
- **广播**: `broadcast_to`, `broadcast_in_dim`
- **连接**: `concat`, `split`, `slice`

## 🎯 里程碑

### Milestone A - 烟雾测试 (≤15个操作)
- ✅ 环境验证
- ✅ 种子集操作配置
- ✅ 小形状网格
- ✅ 100%单个操作编译成功
- ✅ ≥95%成对操作编译成功

### Milestone B - 扩展到60-80个操作
- 🔄 添加更多操作类型
- 🔄 保持编译成功率≥98%
- 🔄 成对操作成功率≥95%

### Milestone C - 达到≥120个操作
- 🔄 完整操作目录覆盖
- 🔄 最终指标记录

## 📝 注意事项

- 从小的形状网格开始，在Milestone A验证后再扩展
- 如果某个方言操作无法合法化，在 `ops.yaml` 中标记为 `unsupported` 并跳过
- 优先使用 `linalg + tensor/arith/math` 以获得最大的合法化覆盖率
- 只包含能够通过提供的传递管道干净地降低的方言

## 🐛 故障排除

### 常见问题

1. **环境检查失败**
   - 安装缺失的工具：`apt install mlir-tools llvm`
   - 安装Python包：`pip3 install jinja2 pyyaml pandas`

2. **MLIR验证失败**
   - 检查模板语法
   - 验证形状参数兼容性

3. **传递管道失败**
   - 检查MLIR版本兼容性
   - 验证传递管道配置

### 日志文件

- 成功编译：`out/single/<op>/logs/<case>.log.txt`
- 编译失败：`out/single/<op>/logs/<case>.err.txt`

## 📈 性能指标

项目跟踪以下关键指标：
- 单个操作编译成功率
- 成对操作编译成功率
- 编译时间分布
- 失败原因分析

## 🤝 贡献

欢迎提交问题和改进建议！

## �� 许可证

本项目采用MIT许可证。
