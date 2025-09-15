#!/usr/bin/env python3
"""
分析IR目录，生成详细的分类统计
"""

import os
import json
from collections import defaultdict, Counter
from pathlib import Path

def analyze_ir_catalog():
    """分析IR目录结构"""
    print("🔍 分析IR目录结构...")
    
    # 统计各个目录的文件数量
    directories = {
        'single': 'out/single',
        'single_llvm': 'out/single_llvm', 
        'single_complete': 'out/single_complete',
        'single_complete_llvm': 'out/single_complete_llvm',
        'pairs': 'out/pairs',
        'pairs_llvm': 'out/pairs_llvm',
        'pairs_complete': 'out/pairs_complete',
        'pairs_complete_llvm': 'out/pairs_complete_llvm'
    }
    
    stats = {}
    total_mlir = 0
    total_llvm = 0
    
    for name, path in directories.items():
        if os.path.exists(path):
            mlir_count = len([f for f in os.listdir(path) if f.endswith('.mlir')]) if os.path.isdir(path) else 0
            llvm_count = len([f for f in os.listdir(path) if f.endswith('.ll')]) if os.path.isdir(path) else 0
            
            # 递归统计子目录
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    mlir_count += len([f for f in files if f.endswith('.mlir')])
                    llvm_count += len([f for f in files if f.endswith('.ll')])
            
            stats[name] = {
                'mlir_files': mlir_count,
                'llvm_files': llvm_count,
                'total_files': mlir_count + llvm_count
            }
            
            if 'mlir' in name:
                total_mlir += mlir_count
            if 'llvm' in name:
                total_llvm += llvm_count
        else:
            stats[name] = {'mlir_files': 0, 'llvm_files': 0, 'total_files': 0}
    
    # 分析操作类型
    print("🔧 分析操作类型...")
    operation_types = defaultdict(int)
    tensor_shapes = defaultdict(int)
    
    # 扫描单个操作
    single_ops = []
    if os.path.exists('out/single'):
        for item in os.listdir('out/single'):
            if os.path.isdir(os.path.join('out/single', item)):
                single_ops.append(item)
    
    # 扫描成对操作
    pair_ops = []
    if os.path.exists('out/pairs_llvm'):
        for item in os.listdir('out/pairs_llvm'):
            if os.path.isdir(os.path.join('out/pairs_llvm', item)):
                pair_ops.append(item)
    
    # 分析tensor形状
    shape_patterns = [
        '1x8x8x8xf32',  # 4D tensor
        '16x16xf32',    # 2D tensor  
        '3x3x8x8xf32',  # 4D kernel
        '1x6x6x8xf32',  # 4D output
        '1x4x4x8xf32',  # 4D output
        '1x8xf32'       # 1D tensor
    ]
    
    for pattern in shape_patterns:
        tensor_shapes[pattern] = 0
    
    # 生成报告
    report = {
        'summary': {
            'total_mlir_files': total_mlir,
            'total_llvm_files': total_llvm,
            'total_files': total_mlir + total_llvm,
            'single_operations': len(single_ops),
            'pair_operations': len(pair_ops)
        },
        'directory_stats': stats,
        'operation_catalog': {
            'single_operations': single_ops,
            'pair_operations': pair_ops[:20]  # 只显示前20个
        },
        'tensor_shapes': dict(tensor_shapes)
    }
    
    # 保存详细报告
    with open('ir_catalog_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 生成Markdown报告
    generate_catalog_report(report)
    
    return report

def generate_catalog_report(report):
    """生成目录报告"""
    with open('IR目录统计报告.md', 'w', encoding='utf-8') as f:
        f.write("# IR目录统计报告\n\n")
        
        # 总体统计
        f.write("## 📊 总体统计\n\n")
        f.write(f"- **总MLIR文件**: {report['summary']['total_mlir_files']:,}\n")
        f.write(f"- **总LLVM文件**: {report['summary']['total_llvm_files']:,}\n")
        f.write(f"- **总文件数**: {report['summary']['total_files']:,}\n")
        f.write(f"- **单个操作数**: {report['summary']['single_operations']}\n")
        f.write(f"- **成对操作数**: {report['summary']['pair_operations']}\n\n")
        
        # 目录统计
        f.write("## 📁 目录统计\n\n")
        f.write("| 目录 | MLIR文件 | LLVM文件 | 总文件数 |\n")
        f.write("|------|----------|----------|----------|\n")
        
        for name, stats in report['directory_stats'].items():
            f.write(f"| {name} | {stats['mlir_files']:,} | {stats['llvm_files']:,} | {stats['total_files']:,} |\n")
        
        f.write("\n")
        
        # 操作类型
        f.write("## 🔧 操作类型\n\n")
        f.write("### 单个操作 (前20个)\n\n")
        for i, op in enumerate(report['operation_catalog']['single_operations'][:20], 1):
            f.write(f"{i}. **{op}**\n")
        
        if len(report['operation_catalog']['single_operations']) > 20:
            f.write(f"\n... 还有 {len(report['operation_catalog']['single_operations']) - 20} 个操作\n")
        
        f.write("\n### 成对操作 (前20个)\n\n")
        for i, op in enumerate(report['operation_catalog']['pair_operations'], 1):
            f.write(f"{i}. **{op}**\n")
        
        if report['summary']['pair_operations'] > 20:
            f.write(f"\n... 还有 {report['summary']['pair_operations'] - 20} 个操作\n")
        
        # 文件大小统计
        f.write("\n## 💾 文件大小统计\n\n")
        f.write("```bash\n")
        f.write("# 各目录大小\n")
        for name in report['directory_stats'].keys():
            if os.path.exists(f"out/{name}"):
                f.write(f"du -sh out/{name}\n")
        f.write("```\n\n")
        
        # 实验建议
        f.write("## 🧪 实验建议\n\n")
        f.write("### 按复杂度分类\n\n")
        f.write("1. **简单操作** (1x8x8x8xf32):\n")
        f.write("   - 元素级运算: add, sub, mul, div\n")
        f.write("   - 激活函数: relu, sigmoid, tanh\n")
        f.write("   - 数学函数: exp, log, sqrt\n\n")
        
        f.write("2. **中等操作** (16x16xf32):\n")
        f.write("   - 矩阵乘法: matmul\n")
        f.write("   - 线性代数运算\n\n")
        
        f.write("3. **复杂操作** (多形状):\n")
        f.write("   - 卷积: conv2d_nhwc_hwcf\n")
        f.write("   - 池化: maxpool2d, avgpool2d\n")
        f.write("   - 成对操作组合\n\n")
        
        f.write("### 实验分组\n\n")
        f.write("1. **基础运算组**: 数学运算 + 激活函数\n")
        f.write("2. **线性代数组**: 矩阵运算 + 卷积\n")
        f.write("3. **组合运算组**: 成对操作测试\n")
        f.write("4. **完整测试组**: 所有操作的综合测试\n\n")
        
        f.write("### 数据格式\n\n")
        f.write("所有IR文件都使用标准格式：\n")
        f.write("- **MLIR**: 使用linalg方言\n")
        f.write("- **LLVM IR**: 标准LLVM IR格式\n")
        f.write("- **输入输出**: 统一的tensor格式\n\n")

def main():
    """主函数"""
    print("🚀 开始分析IR目录...")
    
    # 分析目录
    report = analyze_ir_catalog()
    
    print("✅ 分析完成！")
    print(f"📊 总文件数: {report['summary']['total_files']:,}")
    print(f"🔧 单个操作: {report['summary']['single_operations']}")
    print(f"🔗 成对操作: {report['summary']['pair_operations']}")
    print(f"📋 详细报告: IR目录统计报告.md")
    print(f"📄 数据文件: ir_catalog_analysis.json")

if __name__ == "__main__":
    main()
