# 🚀 快速启动指南

## 1. 环境检查

```bash
bash env/check_env.sh
```

## 2. 生成操作实例

```bash
python3 scripts/enumerate_ops.py
```

## 3. 生成MLIR文件

```bash
python3 scripts/gen_singletons.py
```

## 4. 编译所有操作

```bash
./compile_all.sh
```

## 5. 验证结果

```bash
find out/single -name "*.ll" -type f
```

## 📊 预期结果

- 4个操作实例
- 4个MLIR文件
- 4个LLVM IR文件
- 100%编译成功率

## 🔧 手动编译单个文件

```bash
mlir-opt <input.mlir> \
  --one-shot-bufferize=bufferize-function-boundaries \
  --convert-linalg-to-loops \
  --lower-affine \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --finalize-memref-to-llvm \
  --reconcile-unrealized-casts | \
  mlir-translate --mlir-to-llvmir -o <output.ll>
```

## 📁 关键文件

- `config/ops.yaml` - 操作配置
- `templates/linalg/` - MLIR模板
- `out/single/` - 生成的MLIR和LLVM IR文件
- `compile_all.sh` - 批量编译脚本
