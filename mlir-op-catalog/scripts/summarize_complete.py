#!/usr/bin/env python3
"""
完整操作目录编译结果汇总脚本
生成所有142个操作的编译结果报告
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any

def generate_complete_report():
    """生成完整报告"""
    # 设置路径
    single_llvm_dir = Path("out/single_complete_llvm")
    cases_file = Path("out/cases_single_complete.json")
    
    # 检查文件
    if not single_llvm_dir.exists():
        print("❌ LLVM IR目录不存在")
        return
    
    if not cases_file.exists():
        print("❌ 操作实例文件不存在")
        return
    
    # 加载操作实例
    with open(cases_file, 'r') as f:
        cases = json.load(f)
    
    # 统计LLVM IR文件
    llvm_files = list(single_llvm_dir.rglob("*.ll"))
    
    # 按方言分类统计
    dialect_stats = {}
    for case in cases:
        op_id = case['op_id']
        dialect = op_id.split('.')[0]
        
        if dialect not in dialect_stats:
            dialect_stats[dialect] = {
                'total': 0,
                'compiled': 0,
                'operations': []
            }
        
        dialect_stats[dialect]['total'] += 1
        dialect_stats[dialect]['operations'].append(op_id)
    
    # 检查编译状态
    for dialect, stats in dialect_stats.items():
        for op_id in stats['operations']:
            # 查找对应的LLVM IR文件
            op_dir = single_llvm_dir / op_id
            if op_dir.exists():
                ll_files = list(op_dir.glob("*.ll"))
                if ll_files:
                    stats['compiled'] += 1
    
    # 生成报告
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'summary': {
            'total_operations': len(cases),
            'total_llvm_files': len(llvm_files),
            'compilation_success_rate': len(llvm_files) / len(cases) * 100 if cases else 0
        },
        'dialect_statistics': dialect_stats,
        'compilation_details': {
            'input_directory': 'out/single_complete',
            'output_directory': 'out/single_complete_llvm',
            'pipeline_used': 'config/pipeline_linalg_to_llvm.txt'
        }
    }
    
    # 保存JSON报告
    report_file = Path("out/reports/complete_compilation_report.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 生成Markdown报告
    md_report = generate_markdown_report(report)
    md_file = Path("out/reports/complete_compilation_report.md")
    
    with open(md_file, 'w') as f:
        f.write(md_report)
    
    # 输出到控制台
    print_report(report)
    
    print(f"\n📊 报告已保存到:")
    print(f"  JSON: {report_file}")
    print(f"  Markdown: {md_file}")

def generate_markdown_report(report: Dict[str, Any]) -> str:
    """生成Markdown格式的报告"""
    md = f"""# MLIR完整操作目录编译结果报告

**生成时间**: {report['timestamp']}

## 📊 编译摘要

- **总操作数**: {report['summary']['total_operations']}
- **成功编译**: {report['summary']['total_llvm_files']}
- **编译成功率**: {report['summary']['compilation_success_rate']:.1f}%

## 🔧 编译配置

- **输入目录**: {report['compilation_details']['input_directory']}
- **输出目录**: {report['compilation_details']['output_directory']}
- **编译管道**: {report['compilation_details']['pipeline_used']}

## 📈 按方言统计

"""
    
    for dialect, stats in report['dialect_statistics'].items():
        success_rate = stats['compiled'] / stats['total'] * 100 if stats['total'] > 0 else 0
        md += f"""### {dialect.upper()} 方言

- **操作总数**: {stats['total']}
- **编译成功**: {stats['compiled']}
- **成功率**: {success_rate:.1f}%

**包含操作**:
"""
        for op in stats['operations']:
            md += f"- `{op}`\n"
        md += "\n"
    
    return md

def print_report(report: Dict[str, Any]):
    """打印报告到控制台"""
    print("🎉 MLIR完整操作目录编译结果报告")
    print("=" * 50)
    print(f"📅 生成时间: {report['timestamp']}")
    print()
    print("📊 编译摘要:")
    print(f"  总操作数: {report['summary']['total_operations']}")
    print(f"  成功编译: {report['summary']['total_llvm_files']}")
    print(f"  编译成功率: {report['summary']['compilation_success_rate']:.1f}%")
    print()
    print("📈 按方言统计:")
    
    for dialect, stats in report['dialect_statistics'].items():
        success_rate = stats['compiled'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {dialect.upper()}: {stats['compiled']}/{stats['total']} ({success_rate:.1f}%)")

if __name__ == "__main__":
    generate_complete_report()
