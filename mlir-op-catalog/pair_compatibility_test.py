#!/usr/bin/env python3
"""
成对操作兼容性测试
"""

import os
import json
import subprocess
import time
import statistics
from pathlib import Path
from datetime import datetime
from collections import defaultdict

class PairCompatibilityTest:
    """成对操作兼容性测试类"""
    
    def __init__(self):
        self.results = {
            "test_info": {
                "timestamp": datetime.now().isoformat(),
                "test_type": "pair_compatibility",
                "total_pairs": 0,
                "tested_pairs": 0,
                "compatible_pairs": 0,
                "incompatible_pairs": 0
            },
            "pair_results": {},
            "compatibility_matrix": {},
            "operation_affinity": {},
            "error_analysis": {},
            "recommendations": {}
        }
    
    def test_pair_operation(self, pair_name, llvm_file):
        """测试成对操作"""
        try:
            # 测试LLVM IR语法
            result = subprocess.run(
                ["llvm-as", llvm_file, "-o", "/tmp/test_pair.bc"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                os.remove("/tmp/test_pair.bc")
                return True, None
            else:
                return False, result.stderr
        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except Exception as e:
            return False, str(e)
    
    def analyze_pair_structure(self, pair_name):
        """分析成对操作结构"""
        # 解析操作对
        if "_then_" in pair_name:
            op1, op2 = pair_name.split("_then_", 1)
        else:
            # 处理其他命名格式
            parts = pair_name.split("_")
            if len(parts) >= 2:
                op1 = parts[0]
                op2 = "_".join(parts[1:])
            else:
                op1 = op2 = pair_name
        
        return {
            "operation1": op1,
            "operation2": op2,
            "pair_type": "sequential" if "_then_" in pair_name else "combined"
        }
    
    def test_pair_operations(self, sample_size=20):
        """测试成对操作"""
        print("🔗 测试成对操作兼容性...")
        
        # 获取成对操作列表
        pair_ops = []
        if os.path.exists("out/pairs_llvm"):
            pair_ops = [d for d in os.listdir("out/pairs_llvm") 
                       if os.path.isdir(os.path.join("out/pairs_llvm", d))]
        
        # 随机采样
        import random
        sample_pairs = random.sample(pair_ops, min(sample_size, len(pair_ops)))
        
        for pair_name in sample_pairs:
            print(f"  测试成对操作: {pair_name}")
            
            # 查找LLVM文件
            llvm_dir = f"out/pairs_llvm/{pair_name}"
            llvm_files = []
            if os.path.exists(llvm_dir):
                for root, dirs, files in os.walk(llvm_dir):
                    llvm_files.extend([os.path.join(root, f) for f in files if f.endswith('.ll')])
            
            pair_result = {
                "pair_name": pair_name,
                "structure": self.analyze_pair_structure(pair_name),
                "llvm_files": llvm_files,
                "total_files": len(llvm_files),
                "valid_files": 0,
                "invalid_files": 0,
                "compatibility_score": 0.0,
                "errors": [],
                "test_times": []
            }
            
            # 测试每个LLVM文件
            for llvm_file in llvm_files:
                start_time = time.time()
                is_valid, error = self.test_pair_operation(pair_name, llvm_file)
                end_time = time.time()
                
                pair_result["test_times"].append(end_time - start_time)
                
                if is_valid:
                    pair_result["valid_files"] += 1
                else:
                    pair_result["invalid_files"] += 1
                    pair_result["errors"].append(f"{os.path.basename(llvm_file)}: {error}")
            
            # 计算兼容性分数
            if pair_result["total_files"] > 0:
                pair_result["compatibility_score"] = pair_result["valid_files"] / pair_result["total_files"]
            
            self.results["pair_results"][pair_name] = pair_result
            self.results["test_info"]["tested_pairs"] += 1
            
            if pair_result["compatibility_score"] == 1.0:
                self.results["test_info"]["compatible_pairs"] += 1
            else:
                self.results["test_info"]["incompatible_pairs"] += 1
        
        self.results["test_info"]["total_pairs"] = len(sample_pairs)
    
    def build_compatibility_matrix(self):
        """构建兼容性矩阵"""
        print("📊 构建兼容性矩阵...")
        
        # 统计操作对的出现频率和兼容性
        operation_stats = defaultdict(lambda: {"total_pairs": 0, "compatible_pairs": 0, "avg_score": 0.0})
        
        for pair_name, result in self.results["pair_results"].items():
            structure = result["structure"]
            op1 = structure["operation1"]
            op2 = structure["operation2"]
            score = result["compatibility_score"]
            
            # 统计操作1
            operation_stats[op1]["total_pairs"] += 1
            if score == 1.0:
                operation_stats[op1]["compatible_pairs"] += 1
            operation_stats[op1]["avg_score"] += score
            
            # 统计操作2
            operation_stats[op2]["total_pairs"] += 1
            if score == 1.0:
                operation_stats[op2]["compatible_pairs"] += 1
            operation_stats[op2]["avg_score"] += score
        
        # 计算平均分数
        for op, stats in operation_stats.items():
            if stats["total_pairs"] > 0:
                stats["avg_score"] /= stats["total_pairs"]
                stats["compatibility_rate"] = stats["compatible_pairs"] / stats["total_pairs"]
        
        self.results["compatibility_matrix"] = dict(operation_stats)
    
    def analyze_operation_affinity(self):
        """分析操作亲和性"""
        print("🔍 分析操作亲和性...")
        
        affinity_scores = defaultdict(lambda: defaultdict(float))
        pair_count = defaultdict(lambda: defaultdict(int))
        
        for pair_name, result in self.results["pair_results"].items():
            structure = result["structure"]
            op1 = structure["operation1"]
            op2 = structure["operation2"]
            score = result["compatibility_score"]
            
            # 双向记录亲和性
            affinity_scores[op1][op2] += score
            affinity_scores[op2][op1] += score
            pair_count[op1][op2] += 1
            pair_count[op2][op1] += 1
        
        # 计算平均亲和性分数
        for op1 in affinity_scores:
            for op2 in affinity_scores[op1]:
                if pair_count[op1][op2] > 0:
                    affinity_scores[op1][op2] /= pair_count[op1][op2]
        
        self.results["operation_affinity"] = dict(affinity_scores)
    
    def analyze_errors(self):
        """分析错误模式"""
        print("🔍 分析错误模式...")
        
        error_patterns = defaultdict(int)
        operation_errors = defaultdict(list)
        
        for pair_name, result in self.results["pair_results"].items():
            for error in result["errors"]:
                # 提取错误类型
                if "timeout" in error.lower():
                    error_type = "timeout"
                elif "syntax" in error.lower():
                    error_type = "syntax_error"
                elif "type" in error.lower():
                    error_type = "type_error"
                else:
                    error_type = "other"
                
                error_patterns[error_type] += 1
                
                # 按操作分类错误
                structure = result["structure"]
                operation_errors[structure["operation1"]].append(error_type)
                operation_errors[structure["operation2"]].append(error_type)
        
        self.results["error_analysis"] = {
            "error_patterns": dict(error_patterns),
            "operation_errors": dict(operation_errors)
        }
    
    def generate_recommendations(self):
        """生成兼容性建议"""
        print("💡 生成兼容性建议...")
        
        recommendations = {
            "highly_compatible_operations": [],
            "problematic_operations": [],
            "best_pairs": [],
            "avoid_pairs": [],
            "optimization_suggestions": []
        }
        
        # 找出高兼容性操作
        for op, stats in self.results["compatibility_matrix"].items():
            if stats["compatibility_rate"] >= 0.8:
                recommendations["highly_compatible_operations"].append({
                    "operation": op,
                    "compatibility_rate": stats["compatibility_rate"],
                    "avg_score": stats["avg_score"]
                })
        
        # 找出问题操作
        for op, stats in self.results["compatibility_matrix"].items():
            if stats["compatibility_rate"] < 0.5:
                recommendations["problematic_operations"].append({
                    "operation": op,
                    "compatibility_rate": stats["compatibility_rate"],
                    "avg_score": stats["avg_score"]
                })
        
        # 找出最佳配对
        for pair_name, result in self.results["pair_results"].items():
            if result["compatibility_score"] == 1.0:
                recommendations["best_pairs"].append({
                    "pair": pair_name,
                    "operations": [result["structure"]["operation1"], result["structure"]["operation2"]],
                    "score": result["compatibility_score"]
                })
        
        # 找出应避免的配对
        for pair_name, result in self.results["pair_results"].items():
            if result["compatibility_score"] < 0.5:
                recommendations["avoid_pairs"].append({
                    "pair": pair_name,
                    "operations": [result["structure"]["operation1"], result["structure"]["operation2"]],
                    "score": result["compatibility_score"]
                })
        
        # 优化建议
        if self.results["error_analysis"]["error_patterns"].get("timeout", 0) > 0:
            recommendations["optimization_suggestions"].append("考虑增加超时时间或优化复杂操作")
        
        if self.results["error_analysis"]["error_patterns"].get("syntax_error", 0) > 0:
            recommendations["optimization_suggestions"].append("检查MLIR到LLVM IR的转换管道")
        
        self.results["recommendations"] = recommendations
    
    def generate_report(self):
        """生成兼容性测试报告"""
        report_file = "experiments/results/pair_compatibility_report.json"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        self.generate_markdown_report()
        
        print(f"✅ 兼容性测试报告已生成: {report_file}")
    
    def generate_markdown_report(self):
        """生成Markdown报告"""
        report_file = "experiments/results/pair_compatibility_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 成对操作兼容性测试报告\n\n")
            f.write(f"**测试时间**: {self.results['test_info']['timestamp']}\n")
            f.write(f"**测试类型**: {self.results['test_info']['test_type']}\n")
            f.write(f"**测试对数**: {self.results['test_info']['tested_pairs']}\n")
            f.write(f"**兼容对数**: {self.results['test_info']['compatible_pairs']}\n")
            f.write(f"**不兼容对数**: {self.results['test_info']['incompatible_pairs']}\n")
            f.write(f"**兼容率**: {self.results['test_info']['compatible_pairs']/max(1, self.results['test_info']['tested_pairs'])*100:.1f}%\n\n")
            
            # 兼容性矩阵
            f.write("## 📊 操作兼容性矩阵\n\n")
            f.write("| 操作 | 总配对数 | 兼容配对数 | 兼容率 | 平均分数 |\n")
            f.write("|------|----------|------------|--------|----------|\n")
            
            for op, stats in sorted(self.results["compatibility_matrix"].items()):
                f.write(f"| {op} | {stats['total_pairs']} | {stats['compatible_pairs']} | {stats['compatibility_rate']*100:.1f}% | {stats['avg_score']:.3f} |\n")
            
            # 详细配对结果
            f.write("\n## 🔗 详细配对结果\n\n")
            f.write("| 配对 | 操作1 | 操作2 | 兼容性分数 | 状态 |\n")
            f.write("|------|-------|-------|------------|------|\n")
            
            for pair_name, result in self.results["pair_results"].items():
                structure = result["structure"]
                status = "✅" if result["compatibility_score"] == 1.0 else "❌"
                f.write(f"| {pair_name} | {structure['operation1']} | {structure['operation2']} | {result['compatibility_score']:.3f} | {status} |\n")
            
            # 推荐建议
            f.write("\n## 💡 推荐建议\n\n")
            recommendations = self.results["recommendations"]
            
            f.write("### 高兼容性操作\n\n")
            for op_info in recommendations["highly_compatible_operations"]:
                f.write(f"- **{op_info['operation']}**: 兼容率 {op_info['compatibility_rate']*100:.1f}%\n")
            
            f.write("\n### 问题操作\n\n")
            for op_info in recommendations["problematic_operations"]:
                f.write(f"- **{op_info['operation']}**: 兼容率 {op_info['compatibility_rate']*100:.1f}%\n")
            
            f.write("\n### 最佳配对\n\n")
            for pair_info in recommendations["best_pairs"][:10]:  # 只显示前10个
                f.write(f"- **{pair_info['pair']}**: {pair_info['operations'][0]} + {pair_info['operations'][1]}\n")
            
            f.write("\n### 应避免的配对\n\n")
            for pair_info in recommendations["avoid_pairs"][:10]:  # 只显示前10个
                f.write(f"- **{pair_info['pair']}**: {pair_info['operations'][0]} + {pair_info['operations'][1]} (分数: {pair_info['score']:.3f})\n")
            
            # 错误分析
            f.write("\n## ❌ 错误分析\n\n")
            error_patterns = self.results["error_analysis"]["error_patterns"]
            f.write("| 错误类型 | 出现次数 |\n")
            f.write("|----------|----------|\n")
            
            for error_type, count in error_patterns.items():
                f.write(f"| {error_type} | {count} |\n")

def main():
    """主函数"""
    print("🚀 开始成对操作兼容性测试...")
    
    test = PairCompatibilityTest()
    
    # 测试成对操作
    test.test_pair_operations(sample_size=15)
    
    # 构建兼容性矩阵
    test.build_compatibility_matrix()
    
    # 分析操作亲和性
    test.analyze_operation_affinity()
    
    # 分析错误
    test.analyze_errors()
    
    # 生成建议
    test.generate_recommendations()
    
    # 生成报告
    test.generate_report()
    
    print("✅ 成对操作兼容性测试完成！")
    print(f"📊 测试对数: {test.results['test_info']['tested_pairs']}")
    print(f"✅ 兼容对数: {test.results['test_info']['compatible_pairs']}")
    print(f"❌ 不兼容对数: {test.results['test_info']['incompatible_pairs']}")

if __name__ == "__main__":
    main()
