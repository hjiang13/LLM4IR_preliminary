#!/bin/bash

echo "🔧 批量编译所有MLIR文件到LLVM IR..."

# 编译所有单个操作
find out/single -name "*.mlir" -type f | while read mlir_file; do
    ll_file="${mlir_file%.mlir}.ll"
    echo "编译: $mlir_file -> $ll_file"
    
    mlir-opt "$mlir_file" \
      --one-shot-bufferize=bufferize-function-boundaries \
      --convert-linalg-to-loops \
      --lower-affine \
      --convert-scf-to-cf \
      --convert-cf-to-llvm \
      --convert-arith-to-llvm \
      --convert-func-to-llvm \
      --finalize-memref-to-llvm \
      --reconcile-unrealized-casts | \
      mlir-translate --mlir-to-llvmir -o "$ll_file"
    
    if [ $? -eq 0 ]; then
        echo "✅ 成功: $ll_file"
    else
        echo "❌ 失败: $mlir_file"
    fi
done

echo "🎉 批量编译完成！"
