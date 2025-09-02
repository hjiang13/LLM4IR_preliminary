#!/usr/bin/env python3
"""
成对操作统计脚本
统计所有20,164个成对操作的情况
"""

import json
from pathlib import Path
from typing import Dict, List, Any

def analyze_pairs():
    """分析成对操作"""
    # 设置路径
    cases_file = Path("out/cases_pairs_complete.json")
    pairs_dir = Path("out/pairs_complete")
    
    # 检查文件
    if not cases_file.exists():
        print("❌ 成对操作实例文件不存在")
        return
    
    if not pairs_dir.exists():
        print("❌ 成对操作MLIR目录不存在")
        return
    
    # 加载成对操作实例
    with open(cases_file, 'r') as f:
        cases = json.load(f)
    
    print(f"📊 成对操作分析报告")
    print("=" * 50)
    print(f"📋 总成对操作数: {len(cases)}")
    print()
    
    # 按方言分析
    dialect_pairs = {}
    for case in cases:
        op1_dialect = case['op_a']['op_id'].split('.')[0]
        op2_dialect = case['op_b']['op_id'].split('.')[0]
        
        pair_key = f"{op1_dialect} → {op2_dialect}"
        if pair_key not in dialect_pairs:
            dialect_pairs[pair_key] = 0
        dialect_pairs[pair_key] += 1
    
    # 统计MLIR文件
    mlir_files = list(pairs_dir.rglob("*.mlir"))
    print(f"📁 生成的MLIR文件数: {len(mlir_files)}")
    print()
    
    # 按方言统计
    print("📈 按方言组合统计 (前20个):")
    sorted_pairs = sorted(dialect_pairs.items(), key=lambda x: x[1], reverse=True)
    
    for i, (pair, count) in enumerate(sorted_pairs[:20]):
        print(f"  {i+1:2d}. {pair:<25} : {count:>5} 个")
    
    print()
    
    # 自组合统计
    self_pairs = 0
    for case in cases:
        if case['op_a']['op_id'] == case['op_b']['op_id']:
            self_pairs += 1
    
    print(f"🔄 自组合操作数: {self_pairs}")
    print(f"🔗 跨操作组合数: {len(cases) - self_pairs}")
    print()
    
    # 生成详细报告
    report = {
        'total_pairs': len(cases),
        'total_mlir_files': len(mlir_files),
        'dialect_combinations': dialect_pairs,
        'self_pairs': self_pairs,
        'cross_operation_pairs': len(cases) - self_pairs,
        'top_combinations': sorted_pairs[:20]
    }
    
    # 保存报告
    report_file = Path("out/reports/pairs_analysis_report.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📊 详细报告已保存到: {report_file}")
    
    # 生成Markdown报告
    md_report = generate_markdown_report(report)
    md_file = Path("out/reports/pairs_analysis_report.md")
    
    with open(md_file, 'w') as f:
        f.write(md_report)
    
    print(f"📝 Markdown报告已保存到: {md_file}")

def generate_markdown_report(report: Dict[str, Any]) -> str:
    """生成Markdown格式的报告"""
    md = f"""# MLIR成对操作分析报告

## 📊 总体统计

- **总成对操作数**: {report['total_pairs']:,}
- **生成的MLIR文件数**: {report['total_mlir_files']:,}
- **自组合操作数**: {report['self_pairs']:,}
- **跨操作组合数**: {report['cross_operation_pairs']:,}

## 📈 方言组合统计

### 前20个最常用的组合

"""
    
    for i, (pair, count) in enumerate(report['top_combinations']):
        md += f"{i+1}. **{pair}**: {count:,} 个\n"
    
    md += f"""
### 完整统计

"""
    
    for pair, count in sorted(report['dialect_combinations'].items(), key=lambda x: x[1], reverse=True):
        md += f"- **{pair}**: {count:,} 个\n"
    
    return md

if __name__ == "__main__":
    analyze_pairs()
