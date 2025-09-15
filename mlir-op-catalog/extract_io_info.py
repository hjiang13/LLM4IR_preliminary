#!/usr/bin/env python3
"""
提取MLIR和LLVM IR的输入输出信息
用于后续实验分析
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

def extract_mlir_io_info(mlir_file: str) -> Dict[str, Any]:
    """提取MLIR文件的输入输出信息"""
    try:
        with open(mlir_file, 'r') as f:
            content = f.read()
        
        info = {
            'file': mlir_file,
            'type': 'mlir',
            'inputs': [],
            'outputs': [],
            'operations': []
        }
        
        # 提取函数签名
        func_match = re.search(r'func\.func @main\((.*?)\) -> (.*?) \{', content, re.DOTALL)
        if func_match:
            params = func_match.group(1)
            return_type = func_match.group(2)
            
            # 解析输入参数
            param_pattern = r'%(\w+):\s*(tensor<[^>]+>)'
            for match in re.finditer(param_pattern, params):
                param_name = match.group(1)
                param_type = match.group(2)
                info['inputs'].append({
                    'name': param_name,
                    'type': param_type
                })
            
            # 解析输出类型
            info['outputs'].append({
                'name': 'return',
                'type': return_type.strip()
            })
        
        # 提取操作类型
        op_patterns = [
            r'linalg\.(\w+)',
            r'arith\.(\w+)',
            r'math\.(\w+)',
            r'tensor\.(\w+)',
            r'memref\.(\w+)'
        ]
        
        for pattern in op_patterns:
            for match in re.finditer(pattern, content):
                op = match.group(1)
                if op not in info['operations']:
                    info['operations'].append(op)
        
        return info
        
    except Exception as e:
        return {
            'file': mlir_file,
            'type': 'mlir',
            'error': str(e),
            'inputs': [],
            'outputs': [],
            'operations': []
        }

def extract_llvm_io_info(llvm_file: str) -> Dict[str, Any]:
    """提取LLVM IR文件的输入输出信息"""
    try:
        with open(llvm_file, 'r') as f:
            content = f.read()
        
        info = {
            'file': llvm_file,
            'type': 'llvm',
            'inputs': [],
            'outputs': [],
            'functions': []
        }
        
        # 提取函数定义
        func_pattern = r'define\s+([^{]+)\s+@(\w+)\(([^)]*)\)\s*\{'
        for match in re.finditer(func_pattern, content):
            return_type = match.group(1).strip()
            func_name = match.group(2)
            params = match.group(3)
            
            info['functions'].append({
                'name': func_name,
                'return_type': return_type,
                'parameters': []
            })
            
            # 解析参数
            if params.strip():
                param_pattern = r'(\w+)\s+%(\w+)'
                for param_match in re.finditer(param_pattern, params):
                    param_type = param_match.group(1)
                    param_name = param_match.group(2)
                    info['functions'][-1]['parameters'].append({
                        'name': param_name,
                        'type': param_type
                    })
        
        # 提取输入输出信息（基于tensor结构）
        tensor_pattern = r'tensor<([^>]+)>'
        for match in re.finditer(tensor_pattern, content):
            tensor_shape = match.group(1)
            info['inputs'].append({
                'type': f'tensor<{tensor_shape}>',
                'description': 'tensor input'
            })
        
        return info
        
    except Exception as e:
        return {
            'file': llvm_file,
            'type': 'llvm',
            'error': str(e),
            'inputs': [],
            'outputs': [],
            'functions': []
        }

def scan_directory(directory: str, file_pattern: str) -> List[str]:
    """扫描目录获取文件列表"""
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(file_pattern):
                files.append(os.path.join(root, filename))
    return sorted(files)

def main():
    """主函数"""
    print("🔍 开始提取IR输入输出信息...")
    
    # 扫描所有MLIR和LLVM文件
    mlir_files = []
    llvm_files = []
    
    # 扫描各个目录
    directories = [
        'out/single',
        'out/single_complete', 
        'out/pairs',
        'out/pairs_complete'
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            mlir_files.extend(scan_directory(directory, '.mlir'))
    
    # 扫描LLVM文件
    llvm_directories = [
        'out/single_llvm',
        'out/single_complete_llvm',
        'out/pairs_llvm', 
        'out/pairs_complete_llvm'
    ]
    
    for directory in llvm_directories:
        if os.path.exists(directory):
            llvm_files.extend(scan_directory(directory, '.ll'))
    
    print(f"📊 找到 {len(mlir_files)} 个MLIR文件")
    print(f"📊 找到 {len(llvm_files)} 个LLVM文件")
    
    # 提取信息
    all_info = {
        'summary': {
            'total_mlir_files': len(mlir_files),
            'total_llvm_files': len(llvm_files),
            'total_files': len(mlir_files) + len(llvm_files)
        },
        'mlir_files': [],
        'llvm_files': []
    }
    
    # 处理MLIR文件
    print("🔧 处理MLIR文件...")
    for i, mlir_file in enumerate(mlir_files):
        if i % 100 == 0:
            print(f"   处理进度: {i}/{len(mlir_files)}")
        info = extract_mlir_io_info(mlir_file)
        all_info['mlir_files'].append(info)
    
    # 处理LLVM文件
    print("🔧 处理LLVM文件...")
    for i, llvm_file in enumerate(llvm_files):
        if i % 1000 == 0:
            print(f"   处理进度: {i}/{len(llvm_files)}")
        info = extract_llvm_io_info(llvm_file)
        all_info['llvm_files'].append(info)
    
    # 保存结果
    output_file = 'ir_io_info.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_info, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 信息提取完成！结果保存到: {output_file}")
    
    # 生成统计报告
    generate_summary_report(all_info)

def generate_summary_report(all_info: Dict[str, Any]):
    """生成统计报告"""
    report_file = 'ir_io_summary.md'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# IR输入输出信息统计报告\n\n")
        
        # 总体统计
        f.write("## 📊 总体统计\n\n")
        f.write(f"- **总MLIR文件**: {all_info['summary']['total_mlir_files']}\n")
        f.write(f"- **总LLVM文件**: {all_info['summary']['total_llvm_files']}\n")
        f.write(f"- **总文件数**: {all_info['summary']['total_files']}\n\n")
        
        # MLIR文件统计
        f.write("## 🔧 MLIR文件统计\n\n")
        mlir_ops = {}
        for info in all_info['mlir_files']:
            if 'operations' in info:
                for op in info['operations']:
                    mlir_ops[op] = mlir_ops.get(op, 0) + 1
        
        f.write("### 操作类型分布\n\n")
        for op, count in sorted(mlir_ops.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- **{op}**: {count} 次\n")
        
        # 输入输出类型统计
        f.write("\n### 输入输出类型统计\n\n")
        input_types = {}
        output_types = {}
        
        for info in all_info['mlir_files']:
            for inp in info.get('inputs', []):
                if 'type' in inp:
                    input_types[inp['type']] = input_types.get(inp['type'], 0) + 1
            for out in info.get('outputs', []):
                if 'type' in out:
                    output_types[out['type']] = output_types.get(out['type'], 0) + 1
        
        f.write("#### 输入类型\n\n")
        for t, count in sorted(input_types.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- **{t}**: {count} 次\n")
        
        f.write("\n#### 输出类型\n\n")
        for t, count in sorted(output_types.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- **{t}**: {count} 次\n")
        
        # 示例文件
        f.write("\n## 📝 示例文件\n\n")
        f.write("### MLIR示例\n\n")
        for info in all_info['mlir_files'][:5]:  # 前5个
            f.write(f"**文件**: `{info['file']}`\n")
            f.write(f"- 输入: {len(info.get('inputs', []))} 个\n")
            f.write(f"- 输出: {len(info.get('outputs', []))} 个\n")
            f.write(f"- 操作: {', '.join(info.get('operations', []))}\n\n")
        
        f.write("### LLVM示例\n\n")
        for info in all_info['llvm_files'][:5]:  # 前5个
            f.write(f"**文件**: `{info['file']}`\n")
            f.write(f"- 函数: {len(info.get('functions', []))} 个\n")
            f.write(f"- 输入: {len(info.get('inputs', []))} 个\n\n")
    
    print(f"📋 统计报告已生成: {report_file}")

if __name__ == "__main__":
    main()
