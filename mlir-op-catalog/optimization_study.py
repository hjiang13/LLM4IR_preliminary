#!/usr/bin/env python3
"""
优化研究 - 测试不同MLIR优化传递的效果
"""

import os
import json
import subprocess
import time
import statistics
from pathlib import Path
from datetime import datetime

class OptimizationStudy:
    """优化研究类"""
    
    def __init__(self):
        self.results = {
            "study_info": {
                "timestamp": datetime.now().isoformat(),
                "study_type": "optimization_effectiveness",
                "total_operations": 0,
                "analyzed_operations": 0
            },
            "pass_combinations": {},
            "operation_results": {},
            "optimization_metrics": {},
            "recommendations": {}
        }
        
        # 定义不同的优化传递组合
        self.pass_combinations = {
            "minimal": [
                "convert-linalg-to-loops"
            ],
            "basic": [
                "convert-linalg-to-loops",
                "lower-affine",
                "convert-scf-to-cf"
            ],
            "intermediate": [
                "convert-linalg-to-loops",
                "lower-affine", 
                "convert-scf-to-cf",
                "convert-math-to-llvm",
                "convert-arith-to-llvm"
            ],
            "advanced": [
                "convert-linalg-to-loops",
                "lower-affine",
                "convert-scf-to-cf", 
                "convert-math-to-llvm",
                "convert-arith-to-llvm",
                "convert-func-to-llvm"
            ],
            "full": [
                "one-shot-bufferize=bufferize-function-boundaries",
                "convert-linalg-to-loops",
                "lower-affine",
                "convert-scf-to-cf",
                "convert-math-to-llvm", 
                "convert-arith-to-llvm",
                "convert-func-to-llvm",
                "reconcile-unrealized-casts"
            ]
        }
    
    def test_optimization_passes(self, mlir_file, pass_combination, iterations=5):
        """测试特定优化传递组合"""
        times = []
        success_count = 0
        output_sizes = []
        
        for _ in range(iterations):
            start_time = time.time()
            try:
                cmd = ["mlir-opt", mlir_file] + [f"--{pass_name}" for pass_name in pass_combination]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                end_time = time.time()
                
                if result.returncode == 0:
                    times.append(end_time - start_time)
                    success_count += 1
                    # 估算输出大小（行数）
                    output_sizes.append(len(result.stdout.split('\n')))
            except (subprocess.TimeoutExpired, Exception):
                continue
        
        if times:
            return {
                "success_rate": success_count / iterations,
                "mean_time": statistics.mean(times),
                "median_time": statistics.median(times),
                "min_time": min(times),
                "max_time": max(times),
                "std_time": statistics.stdev(times) if len(times) > 1 else 0,
                "mean_output_size": statistics.mean(output_sizes) if output_sizes else 0,
                "samples": len(times)
            }
        return None
    
    def analyze_operation_optimization(self, operation_name, mlir_file):
        """分析单个操作的优化效果"""
        print(f"  分析操作: {operation_name}")
        
        op_result = {
            "operation": operation_name,
            "mlir_file": mlir_file,
            "pass_results": {},
            "best_pass": None,
            "optimization_effectiveness": {}
        }
        
        # 测试所有传递组合
        for pass_name, passes in self.pass_combinations.items():
            result = self.test_optimization_passes(mlir_file, passes, 3)
            if result:
                op_result["pass_results"][pass_name] = result
        
        # 找出最佳传递组合
        if op_result["pass_results"]:
            best_pass = min(op_result["pass_results"].items(), 
                          key=lambda x: x[1]["mean_time"])
            op_result["best_pass"] = best_pass[0]
            
            # 计算优化效果
            if "minimal" in op_result["pass_results"] and "full" in op_result["pass_results"]:
                minimal_time = op_result["pass_results"]["minimal"]["mean_time"]
                full_time = op_result["pass_results"]["full"]["mean_time"]
                op_result["optimization_effectiveness"] = {
                    "time_improvement": (minimal_time - full_time) / minimal_time * 100,
                    "output_size_change": op_result["pass_results"]["full"]["mean_output_size"] - 
                                        op_result["pass_results"]["minimal"]["mean_output_size"]
                }
        
        return op_result
    
    def run_optimization_study(self, sample_size=10):
        """运行优化研究"""
        print("🔬 开始优化研究...")
        
        # 获取操作列表
        single_ops = []
        if os.path.exists("out/single"):
            single_ops = [d for d in os.listdir("out/single") 
                         if os.path.isdir(os.path.join("out/single", d))]
        
        # 随机采样
        import random
        sample_ops = random.sample(single_ops, min(sample_size, len(single_ops)))
        
        for op in sample_ops:
            mlir_file = f"out/single/{op}/{op}_N1_H8_W8_C8.mlir"
            if os.path.exists(mlir_file):
                op_result = self.analyze_operation_optimization(op, mlir_file)
                self.results["operation_results"][op] = op_result
                self.results["study_info"]["analyzed_operations"] += 1
        
        self.results["study_info"]["total_operations"] = len(sample_ops)
    
    def analyze_pass_effectiveness(self):
        """分析传递效果"""
        print("📊 分析传递效果...")
        
        pass_stats = {}
        
        for pass_name in self.pass_combinations.keys():
            times = []
            success_rates = []
            output_sizes = []
            
            for op, result in self.results["operation_results"].items():
                if pass_name in result["pass_results"]:
                    pass_result = result["pass_results"][pass_name]
                    times.append(pass_result["mean_time"])
                    success_rates.append(pass_result["success_rate"])
                    output_sizes.append(pass_result["mean_output_size"])
            
            if times:
                pass_stats[pass_name] = {
                    "mean_time": statistics.mean(times),
                    "median_time": statistics.median(times),
                    "mean_success_rate": statistics.mean(success_rates),
                    "mean_output_size": statistics.mean(output_sizes),
                    "operations_tested": len(times)
                }
        
        self.results["pass_combinations"] = pass_stats
    
    def generate_optimization_metrics(self):
        """生成优化指标"""
        print("📈 生成优化指标...")
        
        metrics = {
            "time_efficiency": {},
            "success_reliability": {},
            "output_quality": {},
            "overall_ranking": []
        }
        
        # 时间效率排名
        time_ranking = []
        for pass_name, stats in self.results["pass_combinations"].items():
            time_ranking.append((pass_name, stats["mean_time"]))
        time_ranking.sort(key=lambda x: x[1])
        metrics["time_efficiency"] = {rank[0]: i+1 for i, rank in enumerate(time_ranking)}
        
        # 成功率排名
        success_ranking = []
        for pass_name, stats in self.results["pass_combinations"].items():
            success_ranking.append((pass_name, stats["mean_success_rate"]))
        success_ranking.sort(key=lambda x: x[1], reverse=True)
        metrics["success_reliability"] = {rank[0]: i+1 for i, rank in enumerate(success_ranking)}
        
        # 综合排名
        overall_scores = {}
        for pass_name in self.pass_combinations.keys():
            time_rank = metrics["time_efficiency"].get(pass_name, 999)
            success_rank = metrics["success_reliability"].get(pass_name, 999)
            overall_scores[pass_name] = time_rank + success_rank
        
        overall_ranking = sorted(overall_scores.items(), key=lambda x: x[1])
        metrics["overall_ranking"] = [rank[0] for rank in overall_ranking]
        
        self.results["optimization_metrics"] = metrics
    
    def generate_recommendations(self):
        """生成优化建议"""
        print("💡 生成优化建议...")
        
        recommendations = {
            "best_overall": self.results["optimization_metrics"]["overall_ranking"][0],
            "fastest": min(self.results["pass_combinations"].items(), key=lambda x: x[1]["mean_time"])[0],
            "most_reliable": max(self.results["pass_combinations"].items(), key=lambda x: x[1]["mean_success_rate"])[0],
            "use_cases": {
                "development": "basic",
                "production": "advanced", 
                "maximum_optimization": "full",
                "quick_testing": "minimal"
            },
            "operation_specific": {}
        }
        
        # 操作特定建议
        for op, result in self.results["operation_results"].items():
            if result["best_pass"]:
                recommendations["operation_specific"][op] = result["best_pass"]
        
        self.results["recommendations"] = recommendations
    
    def generate_report(self):
        """生成优化研究报告"""
        report_file = "experiments/results/optimization_study_report.json"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        self.generate_markdown_report()
        
        print(f"✅ 优化研究报告已生成: {report_file}")
    
    def generate_markdown_report(self):
        """生成Markdown报告"""
        report_file = "experiments/results/optimization_study_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 优化研究报告\n\n")
            f.write(f"**研究时间**: {self.results['study_info']['timestamp']}\n")
            f.write(f"**研究类型**: {self.results['study_info']['study_type']}\n")
            f.write(f"**分析操作数**: {self.results['study_info']['analyzed_operations']}\n\n")
            
            # 传递组合效果
            f.write("## 🔧 传递组合效果\n\n")
            f.write("| 组合 | 平均时间(ms) | 成功率 | 输出大小 | 排名 |\n")
            f.write("|------|-------------|--------|----------|------|\n")
            
            for pass_name, stats in self.results["pass_combinations"].items():
                time_ms = stats["mean_time"] * 1000
                success_rate = stats["mean_success_rate"] * 100
                output_size = stats["mean_output_size"]
                overall_rank = self.results["optimization_metrics"]["overall_ranking"].index(pass_name) + 1
                
                f.write(f"| {pass_name} | {time_ms:.2f} | {success_rate:.1f}% | {output_size:.0f} | {overall_rank} |\n")
            
            # 优化建议
            f.write("\n## 💡 优化建议\n\n")
            recommendations = self.results["recommendations"]
            
            f.write("### 总体建议\n\n")
            f.write(f"- **最佳综合**: {recommendations['best_overall']}\n")
            f.write(f"- **最快速度**: {recommendations['fastest']}\n")
            f.write(f"- **最高可靠性**: {recommendations['most_reliable']}\n\n")
            
            f.write("### 使用场景建议\n\n")
            for use_case, pass_name in recommendations["use_cases"].items():
                f.write(f"- **{use_case}**: {pass_name}\n")
            
            # 操作特定建议
            f.write("\n### 操作特定建议\n\n")
            f.write("| 操作 | 推荐传递组合 |\n")
            f.write("|------|-------------|\n")
            
            for op, best_pass in recommendations["operation_specific"].items():
                f.write(f"| {op} | {best_pass} |\n")
            
            # 详细分析
            f.write("\n## 📊 详细分析\n\n")
            f.write("### 时间效率排名\n\n")
            for i, pass_name in enumerate(self.results["optimization_metrics"]["overall_ranking"], 1):
                f.write(f"{i}. **{pass_name}**\n")
            
            f.write("\n### 操作优化效果\n\n")
            f.write("| 操作 | 最佳传递 | 时间改进 | 输出变化 |\n")
            f.write("|------|----------|----------|----------|\n")
            
            for op, result in self.results["operation_results"].items():
                best_pass = result["best_pass"] or "N/A"
                time_improvement = result["optimization_effectiveness"].get("time_improvement", 0)
                output_change = result["optimization_effectiveness"].get("output_size_change", 0)
                
                f.write(f"| {op} | {best_pass} | {time_improvement:.1f}% | {output_change:+.0f} |\n")

def main():
    """主函数"""
    print("🚀 开始优化研究...")
    
    study = OptimizationStudy()
    
    # 运行优化研究
    study.run_optimization_study(sample_size=12)
    
    # 分析传递效果
    study.analyze_pass_effectiveness()
    
    # 生成优化指标
    study.generate_optimization_metrics()
    
    # 生成建议
    study.generate_recommendations()
    
    # 生成报告
    study.generate_report()
    
    print("✅ 优化研究完成！")
    print(f"📊 分析操作数: {study.results['study_info']['analyzed_operations']}")

if __name__ == "__main__":
    main()
