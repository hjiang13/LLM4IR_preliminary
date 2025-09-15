#!/usr/bin/env python3
"""
基线测试 - 验证所有操作的基本功能
"""

import os
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime

class BaselineTest:
    """基线测试类"""
    
    def __init__(self):
        self.results = {
            "test_info": {
                "timestamp": datetime.now().isoformat(),
                "test_type": "baseline_validation",
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0
            },
            "single_operations": {},
            "pair_operations": {},
            "errors": []
        }
    
    def test_mlir_syntax(self, mlir_file):
        """测试MLIR语法"""
        try:
            result = subprocess.run(
                ["mlir-opt", mlir_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0, result.stderr
        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except Exception as e:
            return False, str(e)
    
    def test_llvm_syntax(self, llvm_file):
        """测试LLVM IR语法"""
        try:
            result = subprocess.run(
                ["llvm-as", llvm_file, "-o", "/tmp/test.bc"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                os.remove("/tmp/test.bc")
            return result.returncode == 0, result.stderr
        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except Exception as e:
            return False, str(e)
    
    def test_optimization_pipeline(self, mlir_file, passes):
        """测试优化传递管道"""
        try:
            cmd = ["mlir-opt", mlir_file] + passes
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.returncode == 0, result.stderr
        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except Exception as e:
            return False, str(e)
    
    def test_single_operations(self):
        """测试单个操作"""
        print("🔧 测试单个操作...")
        
        # 获取所有单个操作
        single_ops = []
        if os.path.exists("out/single"):
            single_ops = [d for d in os.listdir("out/single") 
                         if os.path.isdir(os.path.join("out/single", d))]
        
        for op in single_ops:
            print(f"  测试操作: {op}")
            
            # 测试MLIR文件
            mlir_file = f"out/single/{op}/{op}_N1_H8_W8_C8.mlir"
            llvm_file = f"out/single_llvm/{op}/{op}_N1_H8_W8_C8/{op}_N1_H8_W8_C8.ll"
            
            op_result = {
                "operation": op,
                "mlir_file": mlir_file,
                "llvm_file": llvm_file,
                "mlir_valid": False,
                "llvm_valid": False,
                "optimization_valid": False,
                "errors": []
            }
            
            # 测试MLIR语法
            if os.path.exists(mlir_file):
                valid, error = self.test_mlir_syntax(mlir_file)
                op_result["mlir_valid"] = valid
                if not valid:
                    op_result["errors"].append(f"MLIR语法错误: {error}")
            else:
                op_result["errors"].append("MLIR文件不存在")
            
            # 测试LLVM IR语法
            if os.path.exists(llvm_file):
                valid, error = self.test_llvm_syntax(llvm_file)
                op_result["llvm_valid"] = valid
                if not valid:
                    op_result["errors"].append(f"LLVM语法错误: {error}")
            else:
                op_result["errors"].append("LLVM文件不存在")
            
            # 测试优化传递
            if op_result["mlir_valid"]:
                passes = ["--convert-linalg-to-loops", "--lower-affine", "--convert-scf-to-cf"]
                valid, error = self.test_optimization_pipeline(mlir_file, passes)
                op_result["optimization_valid"] = valid
                if not valid:
                    op_result["errors"].append(f"优化传递错误: {error}")
            
            self.results["single_operations"][op] = op_result
            
            # 更新统计
            self.results["test_info"]["total_tests"] += 1
            if op_result["mlir_valid"] and op_result["llvm_valid"]:
                self.results["test_info"]["passed_tests"] += 1
            else:
                self.results["test_info"]["failed_tests"] += 1
                self.results["errors"].extend(op_result["errors"])
    
    def test_pair_operations(self, sample_size=10):
        """测试成对操作（采样）"""
        print("🔗 测试成对操作...")
        
        # 获取成对操作列表
        pair_ops = []
        if os.path.exists("out/pairs_llvm"):
            pair_ops = [d for d in os.listdir("out/pairs_llvm") 
                       if os.path.isdir(os.path.join("out/pairs_llvm", d))]
        
        # 随机采样
        import random
        sample_ops = random.sample(pair_ops, min(sample_size, len(pair_ops)))
        
        for op in sample_ops:
            print(f"  测试成对操作: {op}")
            
            # 查找LLVM文件
            llvm_dir = f"out/pairs_llvm/{op}"
            llvm_files = []
            if os.path.exists(llvm_dir):
                for root, dirs, files in os.walk(llvm_dir):
                    llvm_files.extend([os.path.join(root, f) for f in files if f.endswith('.ll')])
            
            op_result = {
                "operation": op,
                "llvm_files": llvm_files,
                "valid_files": 0,
                "total_files": len(llvm_files),
                "errors": []
            }
            
            # 测试每个LLVM文件
            for llvm_file in llvm_files:
                valid, error = self.test_llvm_syntax(llvm_file)
                if valid:
                    op_result["valid_files"] += 1
                else:
                    op_result["errors"].append(f"{llvm_file}: {error}")
            
            self.results["pair_operations"][op] = op_result
            
            # 更新统计
            self.results["test_info"]["total_tests"] += 1
            if op_result["valid_files"] == op_result["total_files"]:
                self.results["test_info"]["passed_tests"] += 1
            else:
                self.results["test_info"]["failed_tests"] += 1
                self.results["errors"].extend(op_result["errors"])
    
    def generate_report(self):
        """生成测试报告"""
        report_file = "experiments/results/baseline_test_report.json"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        self.generate_markdown_report()
        
        print(f"✅ 测试报告已生成: {report_file}")
    
    def generate_markdown_report(self):
        """生成Markdown报告"""
        report_file = "experiments/results/baseline_test_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 基线测试报告\n\n")
            f.write(f"**测试时间**: {self.results['test_info']['timestamp']}\n")
            f.write(f"**测试类型**: {self.results['test_info']['test_type']}\n\n")
            
            # 总体统计
            f.write("## 📊 总体统计\n\n")
            f.write(f"- **总测试数**: {self.results['test_info']['total_tests']}\n")
            f.write(f"- **通过测试**: {self.results['test_info']['passed_tests']}\n")
            f.write(f"- **失败测试**: {self.results['test_info']['failed_tests']}\n")
            f.write(f"- **成功率**: {self.results['test_info']['passed_tests']/max(1, self.results['test_info']['total_tests'])*100:.1f}%\n\n")
            
            # 单个操作统计
            f.write("## 🔧 单个操作测试结果\n\n")
            f.write("| 操作 | MLIR有效 | LLVM有效 | 优化有效 | 状态 |\n")
            f.write("|------|----------|----------|----------|------|\n")
            
            for op, result in self.results['single_operations'].items():
                status = "✅" if result['mlir_valid'] and result['llvm_valid'] else "❌"
                f.write(f"| {op} | {'✅' if result['mlir_valid'] else '❌'} | {'✅' if result['llvm_valid'] else '❌'} | {'✅' if result['optimization_valid'] else '❌'} | {status} |\n")
            
            # 成对操作统计
            f.write("\n## 🔗 成对操作测试结果\n\n")
            f.write("| 操作 | 有效文件 | 总文件 | 成功率 |\n")
            f.write("|------|----------|--------|--------|\n")
            
            for op, result in self.results['pair_operations'].items():
                success_rate = result['valid_files']/max(1, result['total_files'])*100
                f.write(f"| {op} | {result['valid_files']} | {result['total_files']} | {success_rate:.1f}% |\n")
            
            # 错误信息
            if self.results['errors']:
                f.write("\n## ❌ 错误信息\n\n")
                for error in self.results['errors'][:20]:  # 只显示前20个错误
                    f.write(f"- {error}\n")
                if len(self.results['errors']) > 20:
                    f.write(f"\n... 还有 {len(self.results['errors']) - 20} 个错误\n")

def main():
    """主函数"""
    print("🚀 开始基线测试...")
    
    test = BaselineTest()
    
    # 测试单个操作
    test.test_single_operations()
    
    # 测试成对操作（采样）
    test.test_pair_operations(sample_size=10)
    
    # 生成报告
    test.generate_report()
    
    print("✅ 基线测试完成！")
    print(f"📊 总测试数: {test.results['test_info']['total_tests']}")
    print(f"✅ 通过测试: {test.results['test_info']['passed_tests']}")
    print(f"❌ 失败测试: {test.results['test_info']['failed_tests']}")

if __name__ == "__main__":
    main()
