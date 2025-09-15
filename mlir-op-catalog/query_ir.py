#!/usr/bin/env python3
"""
快速查询IR文件信息
"""

import os
import json
import sys
from pathlib import Path

def load_catalog():
    """加载目录信息"""
    try:
        with open('ir_catalog_analysis.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ 未找到目录分析文件，请先运行 analyze_ir_catalog.py")
        return None

def query_operation(operation_name, catalog):
    """查询特定操作的信息"""
    print(f"🔍 查询操作: {operation_name}")
    print("=" * 50)
    
    # 查找单个操作
    single_ops = catalog['operation_catalog']['single_operations']
    if operation_name in single_ops:
        print(f"✅ 找到单个操作: {operation_name}")
        
        # 查找对应的文件
        mlir_file = f"out/single/{operation_name}/{operation_name}_N1_H8_W8_C8.mlir"
        llvm_file = f"out/single_llvm/{operation_name}/{operation_name}_N1_H8_W8_C8/{operation_name}_N1_H8_W8_C8.ll"
        
        if os.path.exists(mlir_file):
            print(f"📄 MLIR文件: {mlir_file}")
        if os.path.exists(llvm_file):
            print(f"📄 LLVM文件: {llvm_file}")
        
        return True
    
    # 查找成对操作
    pair_ops = catalog['operation_catalog']['pair_operations']
    for pair_op in pair_ops:
        if operation_name in pair_op:
            print(f"✅ 找到成对操作: {pair_op}")
            
            # 查找对应的文件
            llvm_file = f"out/pairs_llvm/{pair_op}/{pair_op}/{pair_op}.ll"
            if os.path.exists(llvm_file):
                print(f"📄 LLVM文件: {llvm_file}")
            
            return True
    
    print(f"❌ 未找到操作: {operation_name}")
    return False

def list_operations_by_type(operation_type, catalog):
    """按类型列出操作"""
    print(f"📋 {operation_type}操作列表:")
    print("=" * 50)
    
    single_ops = catalog['operation_catalog']['single_operations']
    
    if operation_type == "数学":
        math_ops = [op for op in single_ops if op in ['add', 'sub', 'mul', 'div', 'pow', 'sqrt', 'exp', 'log']]
        for op in math_ops:
            print(f"  - {op}")
    elif operation_type == "激活":
        activation_ops = [op for op in single_ops if op in ['relu', 'sigmoid', 'tanh', 'gelu', 'swish', 'elu']]
        for op in activation_ops:
            print(f"  - {op}")
    elif operation_type == "线性代数":
        linalg_ops = [op for op in single_ops if op in ['matmul', 'conv2d_nhwc_hwcf', 'maxpool2d', 'avgpool2d']]
        for op in linalg_ops:
            print(f"  - {op}")
    else:
        print("支持的类型: 数学, 激活, 线性代数")

def show_statistics(catalog):
    """显示统计信息"""
    print("📊 目录统计信息:")
    print("=" * 50)
    
    summary = catalog['summary']
    print(f"总文件数: {summary['total_files']:,}")
    print(f"单个操作: {summary['single_operations']}")
    print(f"成对操作: {summary['pair_operations']}")
    
    print("\n📁 目录分布:")
    for name, stats in catalog['directory_stats'].items():
        if stats['total_files'] > 0:
            print(f"  {name}: {stats['total_files']:,} 文件")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python3 query_ir.py <命令> [参数]")
        print("")
        print("命令:")
        print("  query <操作名>     - 查询特定操作")
        print("  list <类型>        - 列出指定类型的操作")
        print("  stats             - 显示统计信息")
        print("  help              - 显示帮助")
        print("")
        print("示例:")
        print("  python3 query_ir.py query add")
        print("  python3 query_ir.py list 数学")
        print("  python3 query_ir.py stats")
        return
    
    catalog = load_catalog()
    if not catalog:
        return
    
    command = sys.argv[1]
    
    if command == "query":
        if len(sys.argv) < 3:
            print("❌ 请指定操作名")
            return
        operation_name = sys.argv[2]
        query_operation(operation_name, catalog)
    
    elif command == "list":
        if len(sys.argv) < 3:
            print("❌ 请指定操作类型")
            return
        operation_type = sys.argv[2]
        list_operations_by_type(operation_type, catalog)
    
    elif command == "stats":
        show_statistics(catalog)
    
    elif command == "help":
        print("IR查询工具帮助")
        print("=============")
        print("")
        print("这个工具可以帮助您快速查询IR文件信息")
        print("")
        print("支持的操作类型:")
        print("  - 数学: add, sub, mul, div, pow, sqrt, exp, log")
        print("  - 激活: relu, sigmoid, tanh, gelu, swish, elu")
        print("  - 线性代数: matmul, conv2d_nhwc_hwcf, maxpool2d, avgpool2d")
        print("")
        print("文件位置:")
        print("  - 单个操作MLIR: out/single/{operation}/")
        print("  - 单个操作LLVM: out/single_llvm/{operation}/")
        print("  - 成对操作LLVM: out/pairs_llvm/{pair_operation}/")
    
    else:
        print(f"❌ 未知命令: {command}")

if __name__ == "__main__":
    main()
