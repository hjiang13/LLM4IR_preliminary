#!/bin/bash

echo "🔗 编译成对操作到LLVM IR..."

# 编译所有成对操作
for mlir_file in out/pairs/*/*.mlir; do
    if [ -f "$mlir_file" ]; then
        ll_file="${mlir_file%.mlir}.ll"
        echo "编译: $mlir_file -> $ll_file"
        
        # Execute the pipeline directly
        mlir-opt "$mlir_file" @config/pipeline_linalg_to_llvm.txt 2> /tmp/err.txt | \
        mlir-translate --mlir-to-llvmir -o "$ll_file" >> /tmp/err.txt 2>&1

        if [ $? -eq 0 ]; then
            echo "✅ 成功: $ll_file"
        else
            echo "❌ 失败: $mlir_file"
        fi
    fi
done

echo "🎉 成对操作编译完成！"
