#!/usr/bin/env python3
"""
性能分析 - 测量不同操作的执行时间
"""

import os
import json
import subprocess
import time
import statistics
from pathlib import Path
from datetime import datetime

class PerformanceAnalysis:
    """性能分析类"""
    
    def __init__(self):
        self.results = {
            "analysis_info": {
                "timestamp": datetime.now().isoformat(),
                "analysis_type": "performance_analysis",
                "total_operations": 0,
                "analyzed_operations": 0
            },
            "single_operations": {},
            "pair_operations": {},
            "optimization_comparison": {},
            "summary_stats": {}
        }
    
    def measure_mlir_compilation_time(self, mlir_file, passes, iterations=10):
        """测量MLIR编译时间"""
        times = []
        
        for _ in range(iterations):
            start_time = time.time()
            try:
                cmd = ["mlir-opt", mlir_file] + passes
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                end_time = time.time()
                
                if result.returncode == 0:
                    times.append(end_time - start_time)
            except (subprocess.TimeoutExpired, Exception):
                continue
        
        if times:
            return {
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "min": min(times),
                "max": max(times),
                "std": statistics.stdev(times) if len(times) > 1 else 0,
                "samples": len(times)
            }
        return None
    
    def measure_llvm_compilation_time(self, llvm_file, iterations=10):
        """测量LLVM编译时间"""
        times = []
        
        for _ in range(iterations):
            start_time = time.time()
            try:
                result = subprocess.run(
                    ["llvm-as", llvm_file, "-o", "/tmp/test.bc"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                end_time = time.time()
                
                if result.returncode == 0:
                    times.append(end_time - start_time)
                    os.remove("/tmp/test.bc")
            except (subprocess.TimeoutExpired, Exception):
                continue
        
        if times:
            return {
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "min": min(times),
                "max": max(times),
                "std": statistics.stdev(times) if len(times) > 1 else 0,
                "samples": len(times)
            }
        return None
    
    def get_file_size(self, file_path):
        """获取文件大小"""
        try:
            return os.path.getsize(file_path)
        except OSError:
            return 0
    
    def analyze_single_operations(self, sample_size=20):
        """分析单个操作性能"""
        print("🔧 分析单个操作性能...")
        
        # 获取所有单个操作
        single_ops = []
        if os.path.exists("out/single"):
            single_ops = [d for d in os.listdir("out/single") 
                         if os.path.isdir(os.path.join("out/single", d))]
        
        # 随机采样
        import random
        sample_ops = random.sample(single_ops, min(sample_size, len(single_ops)))
        
        for op in sample_ops:
            print(f"  分析操作: {op}")
            
            mlir_file = f"out/single/{op}/{op}_N1_H8_W8_C8.mlir"
            llvm_file = f"out/single_llvm/{op}/{op}_N1_H8_W8_C8/{op}_N1_H8_W8_C8.ll"
            
            op_result = {
                "operation": op,
                "mlir_file": mlir_file,
                "llvm_file": llvm_file,
                "mlir_size": 0,
                "llvm_size": 0,
                "compilation_times": {},
                "errors": []
            }
            
            # 文件大小
            if os.path.exists(mlir_file):
                op_result["mlir_size"] = self.get_file_size(mlir_file)
            if os.path.exists(llvm_file):
                op_result["llvm_size"] = self.get_file_size(llvm_file)
            
            # 编译时间测试
            if os.path.exists(mlir_file):
                # 基础优化传递
                basic_passes = ["--convert-linalg-to-loops", "--lower-affine", "--convert-scf-to-cf"]
                basic_time = self.measure_mlir_compilation_time(mlir_file, basic_passes, 5)
                if basic_time:
                    op_result["compilation_times"]["basic"] = basic_time
                
                # 高级优化传递
                advanced_passes = basic_passes + ["--convert-math-to-llvm", "--convert-arith-to-llvm", "--convert-func-to-llvm"]
                advanced_time = self.measure_mlir_compilation_time(mlir_file, advanced_passes, 5)
                if advanced_time:
                    op_result["compilation_times"]["advanced"] = advanced_time
            
            # LLVM编译时间
            if os.path.exists(llvm_file):
                llvm_time = self.measure_llvm_compilation_time(llvm_file, 5)
                if llvm_time:
                    op_result["compilation_times"]["llvm"] = llvm_time
            
            self.results["single_operations"][op] = op_result
            self.results["analysis_info"]["analyzed_operations"] += 1
    
    def analyze_optimization_effectiveness(self):
        """分析优化效果"""
        print("⚡ 分析优化效果...")
        
        optimization_stats = {
            "basic_vs_advanced": {},
            "file_size_impact": {},
            "compilation_time_impact": {}
        }
        
        # 比较基础vs高级优化
        basic_times = []
        advanced_times = []
        
        for op, result in self.results["single_operations"].items():
            if "basic" in result["compilation_times"] and "advanced" in result["compilation_times"]:
                basic_times.append(result["compilation_times"]["basic"]["mean"])
                advanced_times.append(result["compilation_times"]["advanced"]["mean"])
        
        if basic_times and advanced_times:
            optimization_stats["basic_vs_advanced"] = {
                "basic_mean": statistics.mean(basic_times),
                "advanced_mean": statistics.mean(advanced_times),
                "improvement": (statistics.mean(basic_times) - statistics.mean(advanced_times)) / statistics.mean(basic_times) * 100
            }
        
        self.results["optimization_comparison"] = optimization_stats
    
    def generate_summary_statistics(self):
        """生成汇总统计"""
        print("📊 生成汇总统计...")
        
        # 编译时间统计
        all_basic_times = []
        all_advanced_times = []
        all_llvm_times = []
        
        # 文件大小统计
        mlir_sizes = []
        llvm_sizes = []
        
        for op, result in self.results["single_operations"].items():
            if "basic" in result["compilation_times"]:
                all_basic_times.append(result["compilation_times"]["basic"]["mean"])
            if "advanced" in result["compilation_times"]:
                all_advanced_times.append(result["compilation_times"]["advanced"]["mean"])
            if "llvm" in result["compilation_times"]:
                all_llvm_times.append(result["compilation_times"]["llvm"]["mean"])
            
            if result["mlir_size"] > 0:
                mlir_sizes.append(result["mlir_size"])
            if result["llvm_size"] > 0:
                llvm_sizes.append(result["llvm_size"])
        
        summary = {
            "compilation_times": {
                "basic": {
                    "mean": statistics.mean(all_basic_times) if all_basic_times else 0,
                    "median": statistics.median(all_basic_times) if all_basic_times else 0,
                    "min": min(all_basic_times) if all_basic_times else 0,
                    "max": max(all_basic_times) if all_basic_times else 0
                },
                "advanced": {
                    "mean": statistics.mean(all_advanced_times) if all_advanced_times else 0,
                    "median": statistics.median(all_advanced_times) if all_advanced_times else 0,
                    "min": min(all_advanced_times) if all_advanced_times else 0,
                    "max": max(all_advanced_times) if all_advanced_times else 0
                },
                "llvm": {
                    "mean": statistics.mean(all_llvm_times) if all_llvm_times else 0,
                    "median": statistics.median(all_llvm_times) if all_llvm_times else 0,
                    "min": min(all_llvm_times) if all_llvm_times else 0,
                    "max": max(all_llvm_times) if all_llvm_times else 0
                }
            },
            "file_sizes": {
                "mlir": {
                    "mean": statistics.mean(mlir_sizes) if mlir_sizes else 0,
                    "median": statistics.median(mlir_sizes) if mlir_sizes else 0,
                    "min": min(mlir_sizes) if mlir_sizes else 0,
                    "max": max(mlir_sizes) if mlir_sizes else 0
                },
                "llvm": {
                    "mean": statistics.mean(llvm_sizes) if llvm_sizes else 0,
                    "median": statistics.median(llvm_sizes) if llvm_sizes else 0,
                    "min": min(llvm_sizes) if llvm_sizes else 0,
                    "max": max(llvm_sizes) if llvm_sizes else 0
                }
            }
        }
        
        self.results["summary_stats"] = summary
    
    def generate_report(self):
        """生成性能分析报告"""
        report_file = "experiments/results/performance_analysis_report.json"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        self.generate_markdown_report()
        
        print(f"✅ 性能分析报告已生成: {report_file}")
    
    def generate_markdown_report(self):
        """生成Markdown报告"""
        report_file = "experiments/results/performance_analysis_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 性能分析报告\n\n")
            f.write(f"**分析时间**: {self.results['analysis_info']['timestamp']}\n")
            f.write(f"**分析类型**: {self.results['analysis_info']['analysis_type']}\n")
            f.write(f"**分析操作数**: {self.results['analysis_info']['analyzed_operations']}\n\n")
            
            # 汇总统计
            f.write("## 📊 汇总统计\n\n")
            summary = self.results["summary_stats"]
            
            f.write("### 编译时间统计 (秒)\n\n")
            f.write("| 类型 | 平均 | 中位数 | 最小 | 最大 |\n")
            f.write("|------|------|--------|------|------|\n")
            
            for comp_type, stats in summary["compilation_times"].items():
                f.write(f"| {comp_type} | {stats['mean']:.4f} | {stats['median']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |\n")
            
            f.write("\n### 文件大小统计 (字节)\n\n")
            f.write("| 类型 | 平均 | 中位数 | 最小 | 最大 |\n")
            f.write("|------|------|--------|------|------|\n")
            
            for file_type, stats in summary["file_sizes"].items():
                f.write(f"| {file_type} | {stats['mean']:.0f} | {stats['median']:.0f} | {stats['min']:.0f} | {stats['max']:.0f} |\n")
            
            # 优化效果
            if "basic_vs_advanced" in self.results["optimization_comparison"]:
                f.write("\n## ⚡ 优化效果分析\n\n")
                opt_stats = self.results["optimization_comparison"]["basic_vs_advanced"]
                f.write(f"- **基础优化平均时间**: {opt_stats['basic_mean']:.4f}秒\n")
                f.write(f"- **高级优化平均时间**: {opt_stats['advanced_mean']:.4f}秒\n")
                f.write(f"- **性能提升**: {opt_stats['improvement']:.1f}%\n\n")
            
            # 详细操作结果
            f.write("## 🔧 详细操作性能\n\n")
            f.write("| 操作 | 基础编译(ms) | 高级编译(ms) | LLVM编译(ms) | MLIR大小 | LLVM大小 |\n")
            f.write("|------|-------------|-------------|-------------|----------|----------|\n")
            
            for op, result in self.results["single_operations"].items():
                basic_time = result["compilation_times"].get("basic", {}).get("mean", 0) * 1000
                advanced_time = result["compilation_times"].get("advanced", {}).get("mean", 0) * 1000
                llvm_time = result["compilation_times"].get("llvm", {}).get("mean", 0) * 1000
                mlir_size = result["mlir_size"]
                llvm_size = result["llvm_size"]
                
                f.write(f"| {op} | {basic_time:.2f} | {advanced_time:.2f} | {llvm_time:.2f} | {mlir_size} | {llvm_size} |\n")

def main():
    """主函数"""
    print("🚀 开始性能分析...")
    
    analysis = PerformanceAnalysis()
    
    # 分析单个操作
    analysis.analyze_single_operations(sample_size=15)
    
    # 分析优化效果
    analysis.analyze_optimization_effectiveness()
    
    # 生成汇总统计
    analysis.generate_summary_statistics()
    
    # 生成报告
    analysis.generate_report()
    
    print("✅ 性能分析完成！")
    print(f"📊 分析操作数: {analysis.results['analysis_info']['analyzed_operations']}")

if __name__ == "__main__":
    main()
