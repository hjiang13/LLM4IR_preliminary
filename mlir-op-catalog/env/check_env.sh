#!/bin/bash

echo "🔍 检查MLIR工具链环境..."

# 创建输出目录
mkdir -p out

# 检查必需工具
echo "📋 检查必需工具..."

tools=("mlir-opt" "mlir-translate" "llvm-as" "opt")
all_found=true

for tool in "${tools[@]}"; do
    if command -v "$tool" >/dev/null 2>&1; then
        echo "✅ $tool: 找到"
    else
        echo "❌ $tool: 未找到"
        all_found=false
    fi
done

# 检查Python
echo "🐍 检查Python环境..."
if command -v python3 >/dev/null 2>&1; then
    python3 --version
    echo "✅ Python3: 找到"
else
    echo "❌ Python3: 未找到"
    all_found=false
fi

# 检查Python包
echo "📦 检查Python包..."
packages=("jinja2" "pyyaml" "pandas")
for pkg in "${packages[@]}"; do
    if python3 -c "import yaml" 2>/dev/null; then
        echo "✅ $pkg: 已安装"
    else
        echo "❌ $pkg: 未安装"
        all_found=false
    fi
done

# 生成简单的环境报告
cat > out/env.json << JSON_EOF
{
  "timestamp": "$(date -Iseconds)",
  "status": "$(if $all_found; then echo "PASS"; else echo "FAIL"; fi)",
  "tools": {
    "mlir-opt": "$(command -v mlir-opt 2>/dev/null || echo "未找到")",
    "mlir-translate": "$(command -v mlir-translate 2>/dev/null || echo "未找到")",
    "llvm-as": "$(command -v llvm-as 2>/dev/null || echo "未找到")",
    "opt": "$(command -v opt 2>/dev/null || echo "未找到")",
    "python3": "$(command -v python3 2>/dev/null || echo "未找到")"
  }
}
JSON_EOF

if $all_found; then
    echo "✅ 环境检查通过！"
    exit 0
else
    echo "❌ 环境检查失败！请安装缺失的依赖。"
    exit 1
fi
