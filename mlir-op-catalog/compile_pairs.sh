#!/bin/bash

echo "🔗 编译成对操作到LLVM IR..."

# 编译所有成对操作
find out/pairs -name "*.mlir" -type f | while read mlir_file; do
    ll_file="${mlir_file%.mlir}.ll"
    log_dir=$(dirname "$mlir_file")/logs
    mkdir -p "$log_dir"
    log_file="$log_dir/$(basename "$mlir_file" .mlir).log.txt"
    err_file="$log_dir/$(basename "$mlir_file" .mlir).err.txt"

    echo "编译: $mlir_file -> $ll_file"
    start_time=$(python3 -c "import time; print(time.time())")

    # Execute the pipeline directly
    mlir-opt "$mlir_file" @config/pipeline_linalg_to_llvm.txt 2> "$err_file" | \
    mlir-translate --mlir-to-llvmir -o "$ll_file" >> "$err_file" 2>&1

    exit_code=$?
    end_time=$(python3 -c "import time; print(time.time())")
    duration_ms=$(( (end_time - start_time) * 1000 ))

    if [ $exit_code -eq 0 ]; then
        echo "✅ 成功: $ll_file"
        echo "输入文件: $mlir_file" > "$log_file"
        echo "输出文件: $ll_file" >> "$log_file"
        echo "传递管道: linalg" >> "$log_file"
        echo "执行时间: ${duration_ms}ms" >> "$log_file"
        echo "返回码: 0" >> "$log_file"
        echo "成功: True" >> "$log_file"
        cat "$err_file" >> "$log_file"
        rm "$err_file"
    else
        echo "❌ 失败: $mlir_file (返回码: $exit_code)"
        echo "输入文件: $mlir_file" > "$log_file"
        echo "输出文件: $ll_file" >> "$log_file"
        echo "传递管道: linalg" >> "$log_file"
        echo "执行时间: ${duration_ms}ms" >> "$log_file"
        echo "返回码: $exit_code" >> "$log_file"
        echo "成功: False" >> "$log_file"
        cat "$err_file" >> "$log_file"
    fi
done

echo "🎉 成对操作编译完成！"
