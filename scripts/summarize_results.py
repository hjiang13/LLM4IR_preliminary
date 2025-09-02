#!/usr/bin/env python3
"""
结果汇总脚本
汇总单个操作和成对操作的编译结果
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

def load_single_cases(cases_file: str) -> List[Dict[str, Any]]:
    """加载单个操作实例"""
    with open(cases_file, 'r') as f:
        return json.load(f)

def load_pair_cases(cases_file: str) -> List[Dict[str, Any]]:
    """加载成对操作实例"""
    with open(cases_file, 'r') as f:
        return json.load(f)

def check_compilation_status(out_dir: Path, case_type: str) -> Dict[str, Any]:
    """检查编译状态"""
    if case_type == "single":
        base_dir = out_dir / "single"
    else:
        base_dir = out_dir / "pairs"
    
    total_mlir = 0
    total_ll = 0
    success_count = 0
    failed_cases = []
    
    # 统计MLIR文件
    for mlir_file in base_dir.rglob("*.mlir"):
        total_mlir += 1
        case_id = mlir_file.stem
        ll_file = mlir_file.with_suffix('.ll')
        
        if ll_file.exists():
            total_ll += 1
            success_count += 1
        else:
            failed_cases.append(case_id)
    
    return {
        'total_mlir': total_mlir,
        'total_ll': total_ll,
        'success_count': success_count,
        'failed_cases': failed_cases,
        'success_rate': (success_count / total_mlir * 100) if total_mlir > 0 else 0
    }

def generate_summary_report(out_dir: Path) -> Dict[str, Any]:
    """生成汇总报告"""
    summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'single_ops': check_compilation_status(out_dir, "single"),
        'pair_ops': check_compilation_status(out_dir, "pairs"),
        'overall': {}
    }
    
    # 计算总体统计
    total_mlir = summary['single_ops']['total_mlir'] + summary['pair_ops']['total_mlir']
    total_ll = summary['single_ops']['total_ll'] + summary['pair_ops']['total_ll']
    total_success = summary['single_ops']['success_count'] + summary['pair_ops']['success_count']
    
    summary['overall'] = {
        'total_mlir': total_mlir,
        'total_ll': total_ll,
        'total_success': total_success,
        'overall_success_rate': (total_success / total_mlir * 100) if total_mlir > 0 else 0
    }
    
    return summary

def save_summary_report(summary: Dict[str, Any], out_dir: Path):
    """保存汇总报告"""
    # 保存JSON格式
    json_file = out_dir / "reports" / "summary.json"
    json_file.parent.mkdir(exist_ok=True)
    
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 保存Markdown格式
    md_file = out_dir / "reports" / "summary.md"
    with open(md_file, 'w') as f:
        f.write("# MLIR操作目录编译结果汇总\n\n")
        f.write(f"**生成时间**: {summary['timestamp']}\n\n")
        
        f.write("## 📊 总体统计\n\n")
        f.write(f"- **总MLIR文件数**: {summary['overall']['total_mlir']}\n")
        f.write(f"- **总LLVM IR文件数**: {summary['overall']['total_ll']}\n")
        f.write(f"- **总体成功率**: {summary['overall']['overall_success_rate']:.1f}%\n\n")
        
        f.write("## 🔧 单个操作\n\n")
        f.write(f"- **MLIR文件数**: {summary['single_ops']['total_mlir']}\n")
        f.write(f"- **LLVM IR文件数**: {summary['single_ops']['total_ll']}\n")
        f.write(f"- **成功率**: {summary['single_ops']['success_rate']:.1f}%\n\n")
        
        f.write("## 🔗 成对操作\n\n")
        f.write(f"- **MLIR文件数**: {summary['pair_ops']['total_mlir']}\n")
        f.write(f"- **LLVM IR文件数**: {summary['pair_ops']['total_ll']}\n")
        f.write(f"- **成功率**: {summary['pair_ops']['success_rate']:.1f}%\n\n")
        
        if summary['single_ops']['failed_cases']:
            f.write("## ❌ 单个操作失败案例\n\n")
            for case in summary['single_ops']['failed_cases']:
                f.write(f"- {case}\n")
            f.write("\n")
        
        if summary['pair_ops']['failed_cases']:
            f.write("## ❌ 成对操作失败案例\n\n")
            for case in summary['pair_ops']['failed_cases']:
                f.write(f"- {case}\n")
            f.write("\n")
        
        f.write("## 🎯 Milestone A 完成状态\n\n")
        if (summary['single_ops']['success_rate'] >= 100 and 
            summary['pair_ops']['success_rate'] >= 95):
            f.write("✅ **Milestone A 已完成**\n")
            f.write("- 单个操作: 100% 成功率 ✅\n")
            f.write(f"- 成对操作: {summary['pair_ops']['success_rate']:.1f}% 成功率 ✅\n")
        else:
            f.write("❌ **Milestone A 未完成**\n")
            f.write(f"- 单个操作: {summary['single_ops']['success_rate']:.1f}% 成功率 ❌\n")
            f.write(f"- 成对操作: {summary['pair_ops']['success_rate']:.1f}% 成功率 ❌\n")
    
    # 保存CSV格式
    csv_data = []
    csv_data.append(['操作类型', 'MLIR文件数', 'LLVM IR文件数', '成功率'])
    csv_data.append(['单个操作', summary['single_ops']['total_mlir'], 
                    summary['single_ops']['total_ll'], 
                    f"{summary['single_ops']['success_rate']:.1f}%"])
    csv_data.append(['成对操作', summary['pair_ops']['total_mlir'], 
                    summary['pair_ops']['total_ll'], 
                    f"{summary['pair_ops']['success_rate']:.1f}%"])
    csv_data.append(['总计', summary['overall']['total_mlir'], 
                    summary['overall']['total_ll'], 
                    f"{summary['overall']['overall_success_rate']:.1f}%"])
    
    csv_file = out_dir / "reports" / "summary.csv"
    df = pd.DataFrame(csv_data[1:], columns=csv_data[0])
    df.to_csv(csv_file, index=False)
    
    print(f"✅ 汇总报告已保存:")
    print(f"  JSON: {json_file}")
    print(f"  Markdown: {md_file}")
    print(f"  CSV: {csv_file}")

def main():
    """主函数"""
    print("📊 生成结果汇总报告...")
    
    out_dir = Path("out")
    if not out_dir.exists():
        print("❌ 输出目录不存在")
        return
    
    # 生成汇总报告
    summary = generate_summary_report(out_dir)
    
    # 打印汇总信息
    print(f"\n📋 汇总结果:")
    print(f"  单个操作: {summary['single_ops']['total_mlir']} MLIR -> {summary['single_ops']['total_ll']} LLVM IR ({summary['single_ops']['success_rate']:.1f}%)")
    print(f"  成对操作: {summary['pair_ops']['total_mlir']} MLIR -> {summary['pair_ops']['total_ll']} LLVM IR ({summary['pair_ops']['success_rate']:.1f}%)")
    print(f"  总体: {summary['overall']['total_mlir']} MLIR -> {summary['overall']['total_ll']} LLVM IR ({summary['overall']['overall_success_rate']:.1f}%)")
    
    # 保存报告
    save_summary_report(summary, out_dir)
    
    print(f"\n🎉 汇总完成！")

if __name__ == "__main__":
    main()
