#!/bin/bash

echo "🚀 MLIR运行演示脚本"
echo "===================="

# 检查环境
echo "1. 检查MLIR环境..."
if command -v mlir-opt >/dev/null 2>&1; then
    echo "✅ mlir-opt 可用"
else
    echo "❌ mlir-opt 不可用"
    exit 1
fi

# 选择一个简单的操作进行演示
MLIR_FILE="out/single/add/add_N1_H8_W8_C8.mlir"
LLVM_FILE="out/single_llvm/add/add_N1_H8_W8_C8/add_N1_H8_W8_C8.ll"

echo ""
echo "2. 显示MLIR代码..."
echo "文件: $MLIR_FILE"
echo "----------------------------------------"
cat "$MLIR_FILE"
echo ""

echo "3. 验证MLIR代码..."
if mlir-opt "$MLIR_FILE" >/dev/null 2>&1; then
    echo "✅ MLIR代码语法正确"
else
    echo "❌ MLIR代码有语法错误"
    exit 1
fi

echo ""
echo "4. 显示编译后的LLVM IR (前20行)..."
echo "文件: $LLVM_FILE"
echo "----------------------------------------"
head -20 "$LLVM_FILE"
echo "..."

echo ""
echo "5. 验证LLVM IR..."
if llvm-as "$LLVM_FILE" -o /tmp/test.bc 2>/dev/null; then
    echo "✅ LLVM IR语法正确"
    rm -f /tmp/test.bc
else
    echo "❌ LLVM IR有语法错误"
fi

echo ""
echo "6. 运行MLIR优化传递..."
echo "使用传递: convert-linalg-to-loops, lower-affine, convert-scf-to-cf"
mlir-opt "$MLIR_FILE" --convert-linalg-to-loops --lower-affine --convert-scf-to-cf

echo ""
echo "✅ 演示完成！"
echo ""
echo "📝 说明："
echo "- MLIR代码定义了tensor<1x8x8x8xf32>的add操作"
echo "- 使用linalg.generic进行元素级操作"
echo "- 已编译为LLVM IR，可以进行进一步优化和代码生成"
echo ""
echo "🔧 下一步可以："
echo "- 使用mlir-opt进行更多优化"
echo "- 使用mlir-translate转换为LLVM IR"
echo "- 使用llc生成机器码"
echo "- 链接到C/C++程序中使用"
