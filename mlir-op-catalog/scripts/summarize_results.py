#!/usr/bin/env python3
"""
结果汇总脚本
分析编译结果并生成报告
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns

def load_compilation_results(log_dir: Path) -> Dict[str, Any]:
    """加载编译结果"""
    summary_file = log_dir / 'compilation_summary.json'
    
    if not summary_file.exists():
        print(f"❌ 找不到编译结果文件: {summary_file}")
        return {}
    
    with open(summary_file, 'r') as f:
        return json.load(f)

def analyze_single_ops(single_log_dir: Path) -> Dict[str, Any]:
    """分析单个操作的结果"""
    summary_file = single_log_dir / 'compilation_summary.json'
    
    if not summary_file.exists():
        return {}
    
    with open(summary_file, 'r') as f:
        return json.load(f)

def analyze_pairs_ops(pairs_log_dir: Path) -> Dict[str, Any]:
    """分析成对操作的结果"""
    summary_file = pairs_log_dir / 'compilation_summary.json'
    
    if not summary_file.exists():
        return {}
    
    with open(summary_file, 'r') as f:
        return json.load(f)

def generate_operation_stats(results: Dict[str, Any]) -> pd.DataFrame:
    """生成操作统计信息"""
    if not results:
        return pd.DataFrame()
    
    stats = []
    
    for file_path, result in results['results'].items():
        if 'error' in result:
            continue
            
        # 提取操作信息
        file_name = Path(file_path).name
        op_id = file_name.split('_')[0]  # 假设格式: op_id_...mlir
        
        # 统计各步骤的成功情况
        mlir_opt_success = result.get('mlir_opt', {}).get('success', False)
        mlir_translate_success = result.get('mlir_translate', {}).get('success', False)
        llvm_as_success = result.get('llvm_as', {}).get('success', False)
        
        # 统计时间
        mlir_opt_time = result.get('mlir_opt', {}).get('time', 0)
        mlir_translate_time = result.get('mlir_translate', {}).get('time', 0)
        llvm_as_time = result.get('llvm_as', {}).get('time', 0)
        total_time = mlir_opt_time + mlir_translate_time + llvm_as_time
        
        stats.append({
            'file_name': file_name,
            'op_id': op_id,
            'mlir_opt_success': mlir_opt_success,
            'mlir_translate_success': mlir_translate_success,
            'llvm_as_success': llvm_as_success,
            'overall_success': llvm_as_success,
            'mlir_opt_time': mlir_opt_time,
            'mlir_translate_time': mlir_translate_time,
            'llvm_as_time': llvm_as_time,
            'total_time': total_time
        })
    
    return pd.DataFrame(stats)

def generate_summary_report(single_results: Dict[str, Any], pairs_results: Dict[str, Any], output_dir: Path):
    """生成汇总报告"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建报告内容
    report = {
        'summary': {
            'single_operations': {
                'total_files': single_results.get('total_files', 0),
                'success_count': single_results.get('success_count', 0),
                'error_count': single_results.get('error_count', 0),
                'success_rate': single_results.get('success_rate', 0)
            },
            'pair_operations': {
                'total_files': pairs_results.get('total_files', 0),
                'success_count': pairs_results.get('success_count', 0),
                'error_count': pairs_results.get('error_count', 0),
                'success_rate': pairs_results.get('success_rate', 0)
            }
        },
        'milestone_status': {
            'milestone_a': {
                'single_ops_100_percent': single_results.get('success_rate', 0) >= 1.0,
                'pairs_95_percent': pairs_results.get('success_rate', 0) >= 0.95,
                'status': 'PASS' if (single_results.get('success_rate', 0) >= 1.0 and 
                                   pairs_results.get('success_rate', 0) >= 0.95) else 'FAIL'
            }
        }
    }
    
    # 保存报告
    report_file = output_dir / 'summary_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # 生成Markdown报告
    markdown_report = generate_markdown_report(report)
    markdown_file = output_dir / 'summary_report.md'
    with open(markdown_file, 'w') as f:
        f.write(markdown_report)
    
    print(f"📊 报告已生成:")
    print(f"  JSON: {report_file}")
    print(f"  Markdown: {markdown_file}")

def generate_markdown_report(report: Dict[str, Any]) -> str:
    """生成Markdown格式的报告"""
    md = "# MLIR操作目录编译结果汇总报告\n\n"
    
    # 总体摘要
    md += "## 总体摘要\n\n"
    
    single = report['summary']['single_operations']
    pairs = report['summary']['pair_operations']
    
    md += f"- **单个操作**: {single['total_files']} 个文件, 成功 {single['success_count']} 个, 成功率 {single['success_rate']:.2%}\n"
    md += f"- **成对操作**: {pairs['total_files']} 个文件, 成功 {pairs['success_count']} 个, 成功率 {pairs['success_rate']:.2%}\n\n"
    
    # 里程碑状态
    md += "## 里程碑状态\n\n"
    
    milestone = report['milestone_status']['milestone_a']
    md += f"- **Milestone A**: {milestone['status']}\n"
    md += f"  - 单个操作100%成功率: {'✅' if milestone['single_ops_100_percent'] else '❌'}\n"
    md += f"  - 成对操作≥95%成功率: {'✅' if milestone['pairs_95_percent'] else '❌'}\n\n"
    
    # 详细统计
    md += "## 详细统计\n\n"
    
    md += "### 单个操作\n"
    md += f"- 总文件数: {single['total_files']}\n"
    md += f"- 成功数: {single['success_count']}\n"
    md += f"- 失败数: {single['error_count']}\n"
    md += f"- 成功率: {single['success_rate']:.2%}\n\n"
    
    md += "### 成对操作\n"
    md += f"- 总文件数: {pairs['total_files']}\n"
    md += f"- 成功数: {pairs['success_count']}\n"
    md += f"- 失败数: {pairs['error_count']}\n"
    md += f"- 成功率: {pairs['success_rate']:.2%}\n\n"
    
    return md

def create_visualizations(single_df: pd.DataFrame, pairs_df: pd.DataFrame, output_dir: Path):
    """创建可视化图表"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置图表样式
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. 成功率对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 单个操作成功率
    if not single_df.empty:
        success_rate = single_df['overall_success'].mean()
        ax1.pie([success_rate, 1-success_rate], labels=['成功', '失败'], autopct='%1.1f%%')
        ax1.set_title('单个操作成功率')
    
    # 成对操作成功率
    if not pairs_df.empty:
        success_rate = pairs_df['overall_success'].mean()
        ax2.pie([success_rate, 1-success_rate], labels=['成功', '失败'], autopct='%1.1f%%')
        ax2.set_title('成对操作成功率')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'success_rates.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 编译时间分布图
    if not single_df.empty or not pairs_df.empty:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        if not single_df.empty:
            single_df['total_time'].hist(ax=axes[0], bins=20, alpha=0.7)
            axes[0].set_title('单个操作编译时间分布')
            axes[0].set_xlabel('时间 (秒)')
            axes[0].set_ylabel('频次')
        
        if not pairs_df.empty:
            pairs_df['total_time'].hist(ax=axes[1], bins=20, alpha=0.7)
            axes[1].set_title('成对操作编译时间分布')
            axes[1].set_xlabel('时间 (秒)')
            axes[1].set_ylabel('频次')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'compilation_times.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"📊 可视化图表已生成到: {output_dir}")

def main():
    """主函数"""
    # 设置路径
    single_log_dir = Path("out/logs/single")
    pairs_log_dir = Path("out/logs/pairs")
    output_dir = Path("out/reports")
    
    print("📊 分析编译结果...")
    
    # 加载结果
    single_results = analyze_single_ops(single_log_dir)
    pairs_results = analyze_pairs_ops(pairs_log_dir)
    
    if not single_results and not pairs_results:
        print("❌ 未找到任何编译结果")
        return
    
    # 生成统计信息
    single_df = generate_operation_stats(single_results) if single_results else pd.DataFrame()
    pairs_df = generate_operation_stats(pairs_results) if pairs_results else pd.DataFrame()
    
    # 生成报告
    print("📝 生成汇总报告...")
    generate_summary_report(single_results, pairs_results, output_dir)
    
    # 创建可视化
    if not single_df.empty or not pairs_df.empty:
        print("📊 创建可视化图表...")
        create_visualizations(single_df, pairs_df, output_dir)
    
    print("✅ 结果汇总完成！")

if __name__ == "__main__":
    main()
