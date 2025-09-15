#!/usr/bin/env python3
"""
输入-输出-IR数据集生成器
为所有IR文件生成输入数据，运行它们，并记录输出
"""

import os
import json
import numpy as np
import subprocess
import tempfile
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

class InputOutputGenerator:
    """输入输出生成器"""
    
    def __init__(self):
        self.base_dir = Path(".")
        self.output_dir = self.base_dir / "io_dataset"
        self.temp_dir = Path("/tmp/mlir_io_test")
        
        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        
        # 数据集结构
        self.dataset = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_operations": 0,
                "single_operations": 0,
                "pair_operations": 0,
                "successful_tests": 0,
                "failed_tests": 0
            },
            "single_operations": {},
            "pair_operations": {},
            "statistics": {}
        }
        
        # 输入数据配置
        self.input_configs = {
            "tensor_1x8x8x8": {
                "shape": [1, 8, 8, 8],
                "dtype": "float32",
                "range": [-1.0, 1.0]
            },
            "tensor_16x16": {
                "shape": [16, 16],
                "dtype": "float32", 
                "range": [-1.0, 1.0]
            },
            "tensor_32x32": {
                "shape": [32, 32],
                "dtype": "float32",
                "range": [-1.0, 1.0]
            },
            "conv_input": {
                "shape": [1, 8, 8, 8],
                "dtype": "float32",
                "range": [0.0, 1.0]
            },
            "conv_kernel": {
                "shape": [3, 3, 8, 8],
                "dtype": "float32",
                "range": [-0.5, 0.5]
            }
        }
    
    def generate_input_data(self, tensor_config: Dict) -> np.ndarray:
        """生成输入数据"""
        shape = tensor_config["shape"]
        dtype = tensor_config["dtype"]
        min_val, max_val = tensor_config["range"]
        
        # 生成随机数据
        data = np.random.uniform(min_val, max_val, shape).astype(dtype)
        return data
    
    def save_tensor_to_file(self, tensor: np.ndarray, filename: str) -> str:
        """将张量保存到文件"""
        filepath = self.temp_dir / filename
        np.save(filepath, tensor)
        return str(filepath)
    
    def load_tensor_from_file(self, filepath: str) -> np.ndarray:
        """从文件加载张量"""
        return np.load(filepath)
    
    def create_c_wrapper(self, operation_name: str, input_shapes: List[List[int]], 
                        output_shape: List[int], llvm_file: str) -> str:
        """创建C包装器来调用LLVM IR函数"""
        
        # 生成函数签名
        input_params = []
        for i, shape in enumerate(input_shapes):
            total_elements = np.prod(shape)
            input_params.append(f"float* input_{i}")
            input_params.append(f"int64_t* input_{i}_shape")
            input_params.append(f"int64_t* input_{i}_strides")
        
        output_params = [
            "float* output",
            "int64_t* output_shape", 
            "int64_t* output_strides"
        ]
        
        all_params = input_params + output_params
        function_signature = f"void mlir_{operation_name}({', '.join(all_params)})"
        
        c_code = f"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// 外部函数声明
extern void main({', '.join(all_params)});

{function_signature} {{
    // 调用MLIR生成的函数
    main({', '.join([f"input_{i}, input_{i}_shape, input_{i}_strides" for i in range(len(input_shapes))])}, 
         output, output_shape, output_strides);
}}

int main() {{
    // 这里可以添加测试代码
    return 0;
}}
"""
        
        return c_code
    
    def compile_and_run_llvm(self, llvm_file: str, c_wrapper: str, 
                           input_data: List[np.ndarray], operation_name: str) -> Tuple[bool, Any, str]:
        """编译并运行LLVM IR"""
        try:
            # 保存C包装器
            c_file = self.temp_dir / f"{operation_name}_wrapper.c"
            with open(c_file, 'w') as f:
                f.write(c_wrapper)
            
            # 编译LLVM IR到目标文件
            bc_file = self.temp_dir / f"{operation_name}.bc"
            result = subprocess.run(
                ["llvm-as", llvm_file, "-o", str(bc_file)],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                return False, None, f"LLVM assembly failed: {result.stderr}"
            
            # 编译C代码和LLVM IR
            obj_file = self.temp_dir / f"{operation_name}.o"
            result = subprocess.run(
                ["llc", str(bc_file), "-o", str(obj_file)],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                return False, None, f"LLC compilation failed: {result.stderr}"
            
            # 编译C包装器
            wrapper_obj = self.temp_dir / f"{operation_name}_wrapper.o"
            result = subprocess.run(
                ["gcc", "-c", str(c_file), "-o", str(wrapper_obj)],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                return False, None, f"C compilation failed: {result.stderr}"
            
            # 链接生成可执行文件
            exe_file = self.temp_dir / f"{operation_name}_test"
            result = subprocess.run(
                ["gcc", str(obj_file), str(wrapper_obj), "-o", str(exe_file)],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                return False, None, f"Linking failed: {result.stderr}"
            
            # 准备输入数据文件
            input_files = []
            for i, data in enumerate(input_data):
                input_file = self.temp_dir / f"{operation_name}_input_{i}.npy"
                np.save(input_file, data)
                input_files.append(str(input_file))
            
            # 运行程序
            result = subprocess.run(
                [str(exe_file)] + input_files,
                capture_output=True, text=True, timeout=60
            )
            
            if result.returncode != 0:
                return False, None, f"Execution failed: {result.stderr}"
            
            # 读取输出数据
            output_file = self.temp_dir / f"{operation_name}_output.npy"
            if output_file.exists():
                output_data = np.load(output_file)
                return True, output_data, None
            else:
                return False, None, "No output file generated"
                
        except subprocess.TimeoutExpired:
            return False, None, "Execution timeout"
        except Exception as e:
            return False, None, f"Unexpected error: {str(e)}"
    
    def test_single_operation(self, operation_name: str, mlir_file: str, llvm_file: str) -> Dict:
        """测试单个操作"""
        print(f"  测试单个操作: {operation_name}")
        
        result = {
            "operation": operation_name,
            "mlir_file": mlir_file,
            "llvm_file": llvm_file,
            "input_data": [],
            "output_data": None,
            "success": False,
            "error": None,
            "execution_time": 0,
            "input_shapes": [],
            "output_shape": None
        }
        
        try:
            # 解析MLIR文件获取输入输出形状
            input_shapes, output_shape = self.parse_mlir_shapes(mlir_file)
            result["input_shapes"] = input_shapes
            result["output_shape"] = output_shape
            
            # 生成输入数据
            input_data = []
            for shape in input_shapes:
                if len(shape) == 4:  # NHWC格式
                    config = self.input_configs["tensor_1x8x8x8"]
                elif len(shape) == 2:  # 矩阵格式
                    config = self.input_configs["tensor_16x16"]
                else:
                    config = self.input_configs["tensor_1x8x8x8"]
                
                data = self.generate_input_data(config)
                input_data.append(data)
                result["input_data"].append(data.tolist())
            
            # 创建C包装器
            c_wrapper = self.create_c_wrapper(operation_name, input_shapes, output_shape, llvm_file)
            
            # 运行测试
            start_time = time.time()
            success, output_data, error = self.compile_and_run_llvm(
                llvm_file, c_wrapper, input_data, operation_name
            )
            end_time = time.time()
            
            result["success"] = success
            result["execution_time"] = end_time - start_time
            
            if success and output_data is not None:
                result["output_data"] = output_data.tolist()
                result["output_shape"] = list(output_data.shape)
            else:
                result["error"] = error
                
        except Exception as e:
            result["error"] = f"Test failed: {str(e)}"
        
        return result
    
    def parse_mlir_shapes(self, mlir_file: str) -> Tuple[List[List[int]], List[int]]:
        """解析MLIR文件获取输入输出形状"""
        try:
            with open(mlir_file, 'r') as f:
                content = f.read()
            
            # 简单的形状解析（基于已知格式）
            # 这里需要根据实际的MLIR文件格式进行调整
            if "tensor<1x8x8x8xf32>" in content:
                input_shapes = [[1, 8, 8, 8], [1, 8, 8, 8]]
                output_shape = [1, 8, 8, 8]
            elif "tensor<16x16xf32>" in content:
                input_shapes = [[16, 16], [16, 16]]
                output_shape = [16, 16]
            else:
                # 默认形状
                input_shapes = [[1, 8, 8, 8], [1, 8, 8, 8]]
                output_shape = [1, 8, 8, 8]
            
            return input_shapes, output_shape
            
        except Exception as e:
            print(f"Warning: Could not parse shapes from {mlir_file}: {e}")
            return [[1, 8, 8, 8], [1, 8, 8, 8]], [1, 8, 8, 8]
    
    def test_all_single_operations(self, sample_size: int = None):
        """测试所有单个操作"""
        print("🔧 测试所有单个操作...")
        
        # 获取所有单个操作
        single_ops = []
        if os.path.exists("out/single"):
            single_ops = [d for d in os.listdir("out/single") 
                         if os.path.isdir(os.path.join("out/single", d))]
        
        if sample_size:
            import random
            single_ops = random.sample(single_ops, min(sample_size, len(single_ops)))
        
        for op in single_ops:
            mlir_file = f"out/single/{op}/{op}_N1_H8_W8_C8.mlir"
            llvm_file = f"out/single_llvm/{op}/{op}_N1_H8_W8_C8/{op}_N1_H8_W8_C8.ll"
            
            if os.path.exists(mlir_file) and os.path.exists(llvm_file):
                result = self.test_single_operation(op, mlir_file, llvm_file)
                self.dataset["single_operations"][op] = result
                
                if result["success"]:
                    self.dataset["metadata"]["successful_tests"] += 1
                else:
                    self.dataset["metadata"]["failed_tests"] += 1
                    print(f"    ❌ {op}: {result['error']}")
            else:
                print(f"    ⚠️  {op}: 文件不存在")
        
        self.dataset["metadata"]["single_operations"] = len(self.dataset["single_operations"])
    
    def test_all_pair_operations(self, sample_size: int = None):
        """测试所有成对操作"""
        print("🔗 测试所有成对操作...")
        
        # 获取所有成对操作
        pair_ops = []
        if os.path.exists("out/pairs_llvm"):
            pair_ops = [d for d in os.listdir("out/pairs_llvm") 
                       if os.path.isdir(os.path.join("out/pairs_llvm", d))]
        
        if sample_size:
            import random
            pair_ops = random.sample(pair_ops, min(sample_size, len(pair_ops)))
        
        for pair_name in pair_ops:
            print(f"  测试成对操作: {pair_name}")
            
            # 查找LLVM文件
            llvm_dir = f"out/pairs_llvm/{pair_name}"
            llvm_files = []
            if os.path.exists(llvm_dir):
                for root, dirs, files in os.walk(llvm_dir):
                    llvm_files.extend([os.path.join(root, f) for f in files if f.endswith('.ll')])
            
            if llvm_files:
                # 测试第一个LLVM文件
                llvm_file = llvm_files[0]
                result = self.test_single_operation(pair_name, "", llvm_file)
                result["pair_type"] = "sequential" if "_then_" in pair_name else "combined"
                result["llvm_files"] = llvm_files
                
                self.dataset["pair_operations"][pair_name] = result
                
                if result["success"]:
                    self.dataset["metadata"]["successful_tests"] += 1
                else:
                    self.dataset["metadata"]["failed_tests"] += 1
                    print(f"    ❌ {pair_name}: {result['error']}")
            else:
                print(f"    ⚠️  {pair_name}: 未找到LLVM文件")
        
        self.dataset["metadata"]["pair_operations"] = len(self.dataset["pair_operations"])
    
    def generate_statistics(self):
        """生成统计信息"""
        print("📊 生成统计信息...")
        
        stats = {
            "total_operations": len(self.dataset["single_operations"]) + len(self.dataset["pair_operations"]),
            "successful_operations": 0,
            "failed_operations": 0,
            "average_execution_time": 0,
            "input_data_distribution": {},
            "output_data_distribution": {}
        }
        
        all_times = []
        
        # 统计单个操作
        for op, result in self.dataset["single_operations"].items():
            if result["success"]:
                stats["successful_operations"] += 1
                all_times.append(result["execution_time"])
            else:
                stats["failed_operations"] += 1
        
        # 统计成对操作
        for op, result in self.dataset["pair_operations"].items():
            if result["success"]:
                stats["successful_operations"] += 1
                all_times.append(result["execution_time"])
            else:
                stats["failed_operations"] += 1
        
        if all_times:
            stats["average_execution_time"] = sum(all_times) / len(all_times)
        
        self.dataset["statistics"] = stats
    
    def save_dataset(self):
        """保存数据集"""
        print("💾 保存数据集...")
        
        # 保存完整数据集
        dataset_file = self.output_dir / "io_dataset.json"
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(self.dataset, f, indent=2, ensure_ascii=False)
        
        # 保存简化版本（只包含成功的数据）
        simplified_dataset = {
            "metadata": self.dataset["metadata"],
            "successful_single_operations": {
                k: v for k, v in self.dataset["single_operations"].items() 
                if v["success"]
            },
            "successful_pair_operations": {
                k: v for k, v in self.dataset["pair_operations"].items() 
                if v["success"]
            },
            "statistics": self.dataset["statistics"]
        }
        
        simplified_file = self.output_dir / "io_dataset_simplified.json"
        with open(simplified_file, 'w', encoding='utf-8') as f:
            json.dump(simplified_dataset, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        self.generate_markdown_report()
        
        print(f"✅ 数据集已保存:")
        print(f"  📁 完整数据集: {dataset_file}")
        print(f"  📁 简化数据集: {simplified_file}")
    
    def generate_markdown_report(self):
        """生成Markdown报告"""
        report_file = self.output_dir / "io_dataset_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 输入-输出-IR数据集报告\n\n")
            f.write(f"**生成时间**: {self.dataset['metadata']['generated_at']}\n")
            f.write(f"**总操作数**: {self.dataset['metadata']['total_operations']}\n")
            f.write(f"**单个操作数**: {self.dataset['metadata']['single_operations']}\n")
            f.write(f"**成对操作数**: {self.dataset['metadata']['pair_operations']}\n")
            f.write(f"**成功测试数**: {self.dataset['metadata']['successful_tests']}\n")
            f.write(f"**失败测试数**: {self.dataset['metadata']['failed_tests']}\n\n")
            
            # 统计信息
            stats = self.dataset["statistics"]
            f.write("## 📊 统计信息\n\n")
            f.write(f"- **总操作数**: {stats['total_operations']}\n")
            f.write(f"- **成功操作数**: {stats['successful_operations']}\n")
            f.write(f"- **失败操作数**: {stats['failed_operations']}\n")
            f.write(f"- **平均执行时间**: {stats['average_execution_time']:.4f}秒\n\n")
            
            # 成功操作列表
            f.write("## ✅ 成功操作列表\n\n")
            f.write("### 单个操作\n\n")
            for op, result in self.dataset["single_operations"].items():
                if result["success"]:
                    f.write(f"- **{op}**: 执行时间 {result['execution_time']:.4f}秒\n")
            
            f.write("\n### 成对操作\n\n")
            for op, result in self.dataset["pair_operations"].items():
                if result["success"]:
                    f.write(f"- **{op}**: 执行时间 {result['execution_time']:.4f}秒\n")
            
            # 失败操作列表
            failed_ops = []
            for op, result in self.dataset["single_operations"].items():
                if not result["success"]:
                    failed_ops.append((op, result["error"]))
            for op, result in self.dataset["pair_operations"].items():
                if not result["success"]:
                    failed_ops.append((op, result["error"]))
            
            if failed_ops:
                f.write("\n## ❌ 失败操作列表\n\n")
                for op, error in failed_ops:
                    f.write(f"- **{op}**: {error}\n")

def main():
    """主函数"""
    print("🚀 开始生成输入-输出-IR数据集...")
    
    generator = InputOutputGenerator()
    
    # 测试单个操作（采样测试）
    generator.test_all_single_operations(sample_size=10)
    
    # 测试成对操作（采样测试）
    generator.test_all_pair_operations(sample_size=5)
    
    # 生成统计信息
    generator.generate_statistics()
    
    # 保存数据集
    generator.save_dataset()
    
    print("✅ 输入-输出-IR数据集生成完成！")
    print(f"📊 总操作数: {generator.dataset['metadata']['total_operations']}")
    print(f"✅ 成功测试: {generator.dataset['metadata']['successful_tests']}")
    print(f"❌ 失败测试: {generator.dataset['metadata']['failed_tests']}")

if __name__ == "__main__":
    main()
