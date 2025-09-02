# MLIR High-Level Op Catalog → LLVM IR

这个项目旨在构建一个系统，用于生成和编译一个包含≥120个高级MLIR操作的目录到LLVM IR，包括单个操作和所有形状兼容的成对组合。

## 项目结构

```
mlir-op-catalog/
├── env/                    # 环境检查脚本
├── config/                 # 配置文件
│   ├── ops.yaml           # 操作定义
│   ├── dtypes.yaml        # 数据类型配置
│   ├── shapes.yaml        # 形状参数配置
│   └── pipeline_*.txt     # Pass Pipeline配置
├── templates/              # Jinja2模板
│   ├── linalg/            # Linalg操作模板
│   └── common/            # 通用模板
├── scripts/                # Python脚本
├── src/                    # 源代码
├── data/                   # 数据文件
│   └── tf_models/         # TensorFlow模型
└── out/                    # 输出文件
    ├── single/             # 单个操作结果
    ├── pairs/              # 成对操作结果
    ├── logs/               # 日志文件
    └── reports/            # 报告文件
```

## 快速开始

### 1. 环境检查

首先检查您的环境是否满足要求：

```bash
./env/check_env.sh
```

这将验证以下依赖项：
- `mlir-opt`, `mlir-translate`, `llvm-as`, `opt`
- `python3` (≥3.9) 与 `jinja2`, `pyyaml`, `pandas`

### 2. 生成操作实例

生成单个操作和成对操作的实例：

```bash
python3 scripts/enumerate_ops.py
```

### 3. 生成MLIR文件

生成单个操作的MLIR文件：

```bash
python3 scripts/gen_singletons.py
```

生成成对操作的MLIR文件：

```bash
python3 scripts/gen_pairs.py
```

### 4. 编译到LLVM IR

编译单个操作：

```bash
python3 scripts/run_pass_pipeline.py -i out/single -o out/single_llvm -l out/logs/single
```

编译成对操作：

```bash
python3 scripts/run_pass_pipeline.py -i out/pairs -o out/pairs_llvm -l out/logs/pairs
```

### 5. 生成报告

汇总编译结果：

```bash
python3 scripts/summarize_results.py
```

## 配置说明

### 操作配置 (config/ops.yaml)

定义MLIR操作，包括：
- 操作ID和方言
- 输入/输出形状和类型
- 使用的模板文件

### 形状配置 (config/shapes.yaml)

定义形状参数网格：
- 批次大小、空间维度、通道数
- 卷积核大小、输出特征数
- 形状约束和广播规则

### Pass Pipeline (config/pipeline_linalg_to_llvm.txt)

定义从Linalg/Tensor/Memref方言到LLVM IR的转换管道。

## 里程碑

### Milestone A (Smoke Test ≤15 ops)
- [x] 环境验证
- [x] 种子操作集
- [x] 小形状网格
- [x] 单个操作生成/编译 (100%成功率)
- [x] 成对操作生成/编译 (≥95%成功率)

### Milestone B (Expand to 60–80 ops)
- [x] 扩展操作目录
- [x] 维护高编译率

### Milestone C (Reach ≥120 ops)
- [ ] 完成操作目录
- [ ] 记录最终指标

### Milestone D (Optional - TF extraction path)
- [ ] TensorFlow模型转换
- [ ] 操作类型计数

## 输出文件

- **MLIR文件**: `out/single/` 和 `out/pairs/`
- **LLVM IR文件**: 编译后的输出
- **日志文件**: `out/logs/` 包含详细的编译日志
- **报告文件**: `out/reports/` 包含汇总报告和可视化

## 故障排除

### 常见问题

1. **环境检查失败**: 确保安装了所有必需的MLIR/LLVM工具和Python包
2. **编译失败**: 检查Pass Pipeline配置是否与您的MLIR版本兼容
3. **形状不兼容**: 验证 `config/shapes.yaml` 中的约束设置

### 日志分析

详细的编译日志保存在 `out/logs/` 目录中，包括：
- 每个编译步骤的状态
- 执行时间和返回码
- 错误消息和堆栈跟踪

## 贡献

欢迎提交问题和拉取请求来改进这个项目。

## 许可证

[在此添加许可证信息]
