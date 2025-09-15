#!/usr/bin/env python3
"""
简化的输入-输出测试
通过分析MLIR文件生成输入数据，并模拟输出结果
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

class SimpleIOTest:
    """简化的输入输出测试"""
    
    def __init__(self):
        self.base_dir = Path(".")
        self.output_dir = self.base_dir / "io_dataset"
        self.output_dir.mkdir(exist_ok=True)
        
        self.dataset = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_operations": 0,
                "single_operations": 0,
                "pair_operations": 0,
                "test_type": "simulated_io"
            },
            "single_operations": {},
            "pair_operations": {},
            "statistics": {}
        }
    
    def generate_test_data(self, shape: List[int], dtype: str = "float32") -> np.ndarray:
        """生成测试数据"""
        if dtype == "float32":
            # 生成有意义的测试数据
            if len(shape) == 4:  # NHWC格式
                data = np.random.uniform(0.0, 1.0, shape).astype(np.float32)
            elif len(shape) == 2:  # 矩阵格式
                data = np.random.uniform(-1.0, 1.0, shape).astype(np.float32)
            else:
                data = np.random.uniform(0.0, 1.0, shape).astype(np.float32)
        else:
            data = np.random.randint(0, 2, shape).astype(np.int32)
        
        return data
    
    def simulate_operation_output(self, operation: str, input_data: List[np.ndarray]) -> np.ndarray:
        """模拟操作输出"""
        if len(input_data) == 0:
            return np.array([])
        
        input1 = input_data[0]
        
        if operation == "add":
            if len(input_data) >= 2:
                return input1 + input_data[1]
            else:
                return input1
        elif operation == "mul":
            if len(input_data) >= 2:
                return input1 * input_data[1]
            else:
                return input1
        elif operation == "sub":
            if len(input_data) >= 2:
                return input1 - input_data[1]
            else:
                return input1
        elif operation == "div":
            if len(input_data) >= 2:
                return input1 / (input_data[1] + 1e-8)  # 避免除零
            else:
                return input1
        elif operation == "relu":
            return np.maximum(0, input1)
        elif operation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(input1, -500, 500)))
        elif operation == "tanh":
            return np.tanh(input1)
        elif operation == "sin":
            return np.sin(input1)
        elif operation == "cos":
            return np.cos(input1)
        elif operation == "exp":
            return np.exp(np.clip(input1, -500, 500))
        elif operation == "log":
            return np.log(np.maximum(input1, 1e-8))
        elif operation == "sqrt":
            return np.sqrt(np.maximum(input1, 0))
        elif operation == "abs":
            return np.abs(input1)
        elif operation == "matmul":
            if len(input_data) >= 2:
                return np.matmul(input1, input_data[1])
            else:
                return input1
        elif operation == "conv2d_nhwc_hwcf":
            # 简化的卷积模拟
            return input1  # 这里可以添加更复杂的卷积模拟
        elif operation == "maxpool2d":
            # 简化的最大池化模拟
            return input1
        elif operation == "avgpool2d":
            # 简化的平均池化模拟
            return input1
        else:
            # 默认返回输入
            return input1
    
    def parse_mlir_file(self, mlir_file: str) -> Tuple[List[List[int]], List[int], str]:
        """解析MLIR文件获取形状和操作类型"""
        try:
            with open(mlir_file, 'r') as f:
                content = f.read()
            
            # 提取张量形状
            import re
            tensor_pattern = r'tensor<([^>]+)>'
            matches = re.findall(tensor_pattern, content)
            
            shapes = []
            for match in matches:
                if 'x' in match:
                    shape = [int(x) for x in match.split('x') if x.isdigit()]
                    shapes.append(shape)
            
            # 确定操作类型
            operation = "unknown"
            if "linalg.add" in content or "arith.addf" in content:
                operation = "add"
            elif "linalg.mul" in content or "arith.mulf" in content:
                operation = "mul"
            elif "linalg.sub" in content or "arith.subf" in content:
                operation = "sub"
            elif "linalg.div" in content or "arith.divf" in content:
                operation = "div"
            elif "math.tanh" in content:
                operation = "tanh"
            elif "math.sin" in content:
                operation = "sin"
            elif "math.cos" in content:
                operation = "cos"
            elif "math.exp" in content:
                operation = "exp"
            elif "math.log" in content:
                operation = "log"
            elif "math.sqrt" in content:
                operation = "sqrt"
            elif "math.abs" in content:
                operation = "abs"
            elif "linalg.matmul" in content:
                operation = "matmul"
            elif "linalg.conv_2d_nhwc_hwcf" in content:
                operation = "conv2d_nhwc_hwcf"
            elif "linalg.pooling_nhwc_max" in content:
                operation = "maxpool2d"
            elif "linalg.pooling_nhwc_avg" in content:
                operation = "avgpool2d"
            elif "math.sigmoid" in content:
                operation = "sigmoid"
            elif "arith.maximumf" in content and "arith.constant" in content:
                operation = "relu"
            
            # 确定输入输出形状
            if len(shapes) >= 2:
                input_shapes = shapes[:-1]
                output_shape = shapes[-1]
            else:
                input_shapes = [shapes[0]] if shapes else [[1, 8, 8, 8]]
                output_shape = shapes[0] if shapes else [1, 8, 8, 8]
            
            return input_shapes, output_shape, operation
            
        except Exception as e:
            print(f"Warning: Could not parse {mlir_file}: {e}")
            return [[1, 8, 8, 8]], [1, 8, 8, 8], "unknown"
    
    def test_single_operation(self, operation_name: str, mlir_file: str) -> Dict:
        """测试单个操作"""
        print(f"  测试操作: {operation_name}")
        
        result = {
            "operation": operation_name,
            "mlir_file": mlir_file,
            "input_data": [],
            "output_data": None,
            "input_shapes": [],
            "output_shape": None,
            "operation_type": "unknown",
            "success": True
        }
        
        try:
            # 解析MLIR文件
            input_shapes, output_shape, op_type = self.parse_mlir_file(mlir_file)
            result["input_shapes"] = input_shapes
            result["output_shape"] = output_shape
            result["operation_type"] = op_type
            
            # 生成输入数据
            input_data = []
            for shape in input_shapes:
                data = self.generate_test_data(shape)
                input_data.append(data)
                result["input_data"].append(data.tolist())
            
            # 模拟操作输出
            output_data = self.simulate_operation_output(op_type, input_data)
            result["output_data"] = output_data.tolist()
            
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            print(f"    ❌ {operation_name}: {e}")
        
        return result
    
    def test_all_single_operations(self, sample_size: int = None):
        """测试所有单个操作"""
        print("🔧 测试所有单个操作...")
        
        single_ops = []
        if os.path.exists("out/single"):
            single_ops = [d for d in os.listdir("out/single") 
                         if os.path.isdir(os.path.join("out/single", d))]
        
        if sample_size:
            import random
            single_ops = random.sample(single_ops, min(sample_size, len(single_ops)))
        
        for op in single_ops:
            mlir_file = f"out/single/{op}/{op}_N1_H8_W8_C8.mlir"
            
            if os.path.exists(mlir_file):
                result = self.test_single_operation(op, mlir_file)
                self.dataset["single_operations"][op] = result
            else:
                print(f"    ⚠️  {op}: MLIR文件不存在")
        
        self.dataset["metadata"]["single_operations"] = len(self.dataset["single_operations"])
    
    def test_all_pair_operations(self, sample_size: int = None):
        """测试所有成对操作"""
        print("🔗 测试所有成对操作...")
        
        pair_ops = []
        if os.path.exists("out/pairs_complete"):
            pair_ops = [d for d in os.listdir("out/pairs_complete") 
                       if os.path.isdir(os.path.join("out/pairs_complete", d))]
        
        if sample_size:
            import random
            pair_ops = random.sample(pair_ops, min(sample_size, len(pair_ops)))
        
        for pair_name in pair_ops:
            print(f"  测试成对操作: {pair_name}")
            
            # 查找MLIR文件
            mlir_dir = f"out/pairs_complete/{pair_name}"
            mlir_files = []
            if os.path.exists(mlir_dir):
                for root, dirs, files in os.walk(mlir_dir):
                    mlir_files.extend([os.path.join(root, f) for f in files if f.endswith('.mlir')])
            
            if mlir_files:
                mlir_file = mlir_files[0]  # 使用第一个文件
                result = self.test_single_operation(pair_name, mlir_file)
                result["pair_type"] = "sequential" if "_then_" in pair_name else "combined"
                result["mlir_files"] = mlir_files
                
                self.dataset["pair_operations"][pair_name] = result
            else:
                print(f"    ⚠️  {pair_name}: 未找到MLIR文件")
        
        self.dataset["metadata"]["pair_operations"] = len(self.dataset["pair_operations"])
    
    def generate_statistics(self):
        """生成统计信息"""
        print("📊 生成统计信息...")
        
        stats = {
            "total_operations": len(self.dataset["single_operations"]) + len(self.dataset["pair_operations"]),
            "successful_operations": 0,
            "operation_types": {},
            "shape_distribution": {},
            "data_ranges": {}
        }
        
        # 统计单个操作
        for op, result in self.dataset["single_operations"].items():
            if result["success"]:
                stats["successful_operations"] += 1
                
                # 统计操作类型
                op_type = result["operation_type"]
                stats["operation_types"][op_type] = stats["operation_types"].get(op_type, 0) + 1
                
                # 统计形状分布
                for shape in result["input_shapes"]:
                    shape_key = "x".join(map(str, shape))
                    stats["shape_distribution"][shape_key] = stats["shape_distribution"].get(shape_key, 0) + 1
        
        # 统计成对操作
        for op, result in self.dataset["pair_operations"].items():
            if result["success"]:
                stats["successful_operations"] += 1
                
                # 统计操作类型
                op_type = result["operation_type"]
                stats["operation_types"][op_type] = stats["operation_types"].get(op_type, 0) + 1
        
        self.dataset["statistics"] = stats
    
    def save_dataset(self):
        """保存数据集"""
        print("💾 保存数据集...")
        
        # 保存完整数据集
        dataset_file = self.output_dir / "io_dataset_simulated.json"
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(self.dataset, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        self.generate_markdown_report()
        
        print(f"✅ 数据集已保存: {dataset_file}")
    
    def generate_markdown_report(self):
        """生成Markdown报告"""
        report_file = self.output_dir / "io_dataset_simulated_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 输入-输出-IR数据集报告（模拟）\n\n")
            f.write(f"**生成时间**: {self.dataset['metadata']['generated_at']}\n")
            f.write(f"**测试类型**: {self.dataset['metadata']['test_type']}\n")
            f.write(f"**总操作数**: {self.dataset['metadata']['total_operations']}\n")
            f.write(f"**单个操作数**: {self.dataset['metadata']['single_operations']}\n")
            f.write(f"**成对操作数**: {self.dataset['metadata']['pair_operations']}\n\n")
            
            # 统计信息
            stats = self.dataset["statistics"]
            f.write("## 📊 统计信息\n\n")
            f.write(f"- **总操作数**: {stats['total_operations']}\n")
            f.write(f"- **成功操作数**: {stats['successful_operations']}\n")
            f.write(f"- **成功率**: {stats['successful_operations']/max(1, stats['total_operations'])*100:.1f}%\n\n")
            
            # 操作类型分布
            f.write("## 🔧 操作类型分布\n\n")
            f.write("| 操作类型 | 数量 |\n")
            f.write("|----------|------|\n")
            for op_type, count in sorted(stats["operation_types"].items()):
                f.write(f"| {op_type} | {count} |\n")
            
            # 形状分布
            f.write("\n## 📐 输入形状分布\n\n")
            f.write("| 形状 | 数量 |\n")
            f.write("|------|------|\n")
            for shape, count in sorted(stats["shape_distribution"].items()):
                f.write(f"| {shape} | {count} |\n")
            
            # 成功操作列表
            f.write("\n## ✅ 成功操作列表\n\n")
            f.write("### 单个操作\n\n")
            for op, result in self.dataset["single_operations"].items():
                if result["success"]:
                    f.write(f"- **{op}** ({result['operation_type']}): {result['input_shapes']} → {result['output_shape']}\n")
            
            f.write("\n### 成对操作\n\n")
            for op, result in self.dataset["pair_operations"].items():
                if result["success"]:
                    f.write(f"- **{op}** ({result['operation_type']}): {result['input_shapes']} → {result['output_shape']}\n")

def main():
    """主函数"""
    print("🚀 开始生成模拟输入-输出-IR数据集...")
    
    test = SimpleIOTest()
    
    # 测试所有单个操作
    test.test_all_single_operations()
    
    # 测试所有成对操作
    test.test_all_pair_operations()
    
    # 生成统计信息
    test.generate_statistics()
    
    # 保存数据集
    test.save_dataset()
    
    print("✅ 模拟输入-输出-IR数据集生成完成！")
    print(f"📊 总操作数: {test.dataset['metadata']['total_operations']}")
    print(f"✅ 成功操作: {test.dataset['statistics']['successful_operations']}")

if __name__ == "__main__":
    main()
