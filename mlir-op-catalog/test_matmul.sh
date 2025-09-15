#!/bin/bash

echo "🧮 测试矩阵乘法MLIR操作"
echo "======================="

MATMUL_MLIR="out/single/matmul/matmul_M16_K16_N16.mlir"
MATMUL_LLVM="out/single_llvm/matmul/matmul_M16_K16_N16/matmul_M16_K16_N16.ll"

echo "1. 显示矩阵乘法MLIR代码..."
echo "文件: $MATMUL_MLIR"
echo "----------------------------------------"
cat "$MATMUL_MLIR"
echo ""

echo "2. 验证MLIR代码..."
if mlir-opt "$MATMUL_MLIR" >/dev/null 2>&1; then
    echo "✅ 矩阵乘法MLIR代码语法正确"
else
    echo "❌ 矩阵乘法MLIR代码有语法错误"
    exit 1
fi

echo ""
echo "3. 运行MLIR优化传递..."
echo "使用传递: convert-linalg-to-loops, lower-affine, convert-scf-to-cf"
mlir-opt "$MATMUL_MLIR" --convert-linalg-to-loops --lower-affine --convert-scf-to-cf

echo ""
echo "4. 显示编译后的LLVM IR (前30行)..."
echo "文件: $MATMUL_LLVM"
echo "----------------------------------------"
head -30 "$MATMUL_LLVM"
echo "..."

echo ""
echo "5. 验证LLVM IR..."
if llvm-as "$MATMUL_LLVM" -o /tmp/matmul_test.bc 2>/dev/null; then
    echo "✅ 矩阵乘法LLVM IR语法正确"
    rm -f /tmp/matmul_test.bc
else
    echo "❌ 矩阵乘法LLVM IR有语法错误"
fi

echo ""
echo "6. 测试其他操作..."
echo "测试ReLU激活函数..."

RELU_MLIR="out/single/relu/relu_N1_H8_W8_C8.mlir"
if [ -f "$RELU_MLIR" ]; then
    echo "ReLU MLIR代码:"
    cat "$RELU_MLIR"
    echo ""
    if mlir-opt "$RELU_MLIR" >/dev/null 2>&1; then
        echo "✅ ReLU MLIR代码语法正确"
    else
        echo "❌ ReLU MLIR代码有语法错误"
    fi
else
    echo "❌ ReLU文件不存在"
fi

echo ""
echo "✅ 矩阵乘法测试完成！"
echo ""
echo "📊 项目统计："
echo "- 单个操作数量: $(find out/single -name "*.mlir" | wc -l)"
echo "- 已编译LLVM文件数量: $(find out/single_llvm -name "*.ll" | wc -l)"
echo "- 成对操作数量: $(find out/pairs_llvm -name "*.ll" | wc -l)"
