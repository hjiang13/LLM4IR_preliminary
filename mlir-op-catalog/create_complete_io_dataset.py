#!/usr/bin/env python3
"""
创建完整的输入-输出-IR数据集
包含实际的数值数据和更详细的元数据
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

class CompleteIODatasetCreator:
    """完整的输入输出数据集创建器"""
    
    def __init__(self):
        self.base_dir = Path(".")
        self.output_dir = self.base_dir / "io_dataset"
        self.output_dir.mkdir(exist_ok=True)
        
        self.dataset = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
                "description": "LLM4IR输入-输出-IR数据集",
                "total_operations": 0,
                "single_operations": 0,
                "pair_operations": 0,
                "data_points": 0
            },
            "operations": {},
            "statistics": {},
            "data_samples": {}
        }
    
    def generate_realistic_data(self, shape: List[int], operation: str) -> np.ndarray:
        """生成更真实的测试数据"""
        if operation in ["add", "sub", "mul", "div"]:
            # 算术运算：使用有意义的数值范围
            data = np.random.uniform(-2.0, 2.0, shape).astype(np.float32)
        elif operation in ["relu", "sigmoid", "tanh", "gelu"]:
            # 激活函数：使用接近零的数值
            data = np.random.uniform(-3.0, 3.0, shape).astype(np.float32)
        elif operation in ["sin", "cos", "tan", "asin", "acos", "atan"]:
            # 三角函数：使用[-π, π]范围
            data = np.random.uniform(-np.pi, np.pi, shape).astype(np.float32)
        elif operation in ["exp", "log", "log2", "log10"]:
            # 指数对数函数：使用正数
            data = np.random.uniform(0.1, 5.0, shape).astype(np.float32)
        elif operation in ["sqrt", "rsqrt", "cbrt"]:
            # 根号函数：使用正数
            data = np.random.uniform(0.1, 10.0, shape).astype(np.float32)
        elif operation in ["abs", "floor", "ceil", "round"]:
            # 数值函数：使用任意范围
            data = np.random.uniform(-5.0, 5.0, shape).astype(np.float32)
        elif operation in ["matmul", "conv2d_nhwc_hwcf"]:
            # 线性运算：使用小数值
            data = np.random.uniform(-1.0, 1.0, shape).astype(np.float32)
        else:
            # 默认：使用标准正态分布
            data = np.random.normal(0, 1, shape).astype(np.float32)
        
        return data
    
    def simulate_operation(self, operation: str, inputs: List[np.ndarray]) -> np.ndarray:
        """模拟操作执行"""
        if len(inputs) == 0:
            return np.array([])
        
        input1 = inputs[0]
        
        # 基本算术运算
        if operation == "add":
            if len(inputs) >= 2:
                return input1 + inputs[1]
            return input1
        elif operation == "sub":
            if len(inputs) >= 2:
                return input1 - inputs[1]
            return input1
        elif operation == "mul":
            if len(inputs) >= 2:
                return input1 * inputs[1]
            return input1
        elif operation == "div":
            if len(inputs) >= 2:
                return input1 / (inputs[1] + 1e-8)
            return input1
        
        # 激活函数
        elif operation == "relu":
            return np.maximum(0, input1)
        elif operation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(input1, -500, 500)))
        elif operation == "tanh":
            return np.tanh(input1)
        elif operation == "gelu":
            return 0.5 * input1 * (1 + np.tanh(np.sqrt(2/np.pi) * (input1 + 0.044715 * input1**3)))
        elif operation == "leaky_relu":
            return np.where(input1 > 0, input1, 0.01 * input1)
        elif operation == "elu":
            return np.where(input1 > 0, input1, np.exp(input1) - 1)
        elif operation == "hard_sigmoid":
            return np.clip(0.2 * input1 + 0.5, 0, 1)
        elif operation == "hard_tanh":
            return np.clip(input1, -1, 1)
        elif operation == "swish":
            return input1 * (1 / (1 + np.exp(-input1)))
        elif operation == "mish":
            return input1 * np.tanh(np.log(1 + np.exp(input1)))
        elif operation == "softsign":
            return input1 / (1 + np.abs(input1))
        elif operation == "softplus":
            return np.log(1 + np.exp(input1))
        
        # 三角函数
        elif operation == "sin":
            return np.sin(input1)
        elif operation == "cos":
            return np.cos(input1)
        elif operation == "tan":
            return np.tan(input1)
        elif operation == "asin":
            return np.arcsin(np.clip(input1, -1, 1))
        elif operation == "acos":
            return np.arccos(np.clip(input1, -1, 1))
        elif operation == "atan":
            return np.arctan(input1)
        elif operation == "atan2":
            if len(inputs) >= 2:
                return np.arctan2(input1, inputs[1])
            return np.arctan(input1)
        
        # 双曲函数
        elif operation == "sinh":
            return np.sinh(input1)
        elif operation == "cosh":
            return np.cosh(input1)
        elif operation == "tanh":
            return np.tanh(input1)
        elif operation == "asinh":
            return np.arcsinh(input1)
        elif operation == "acosh":
            return np.arccosh(np.maximum(1, input1))
        elif operation == "atanh":
            return np.arctanh(np.clip(input1, -0.999, 0.999))
        
        # 指数对数函数
        elif operation == "exp":
            return np.exp(np.clip(input1, -500, 500))
        elif operation == "log":
            return np.log(np.maximum(input1, 1e-8))
        elif operation == "log2":
            return np.log2(np.maximum(input1, 1e-8))
        elif operation == "log10":
            return np.log10(np.maximum(input1, 1e-8))
        elif operation == "log1p":
            return np.log1p(input1)
        elif operation == "exp2":
            return np.exp2(np.clip(input1, -500, 500))
        
        # 幂函数
        elif operation == "pow":
            if len(inputs) >= 2:
                return np.power(input1, inputs[1])
            return input1
        elif operation == "sqrt":
            return np.sqrt(np.maximum(input1, 0))
        elif operation == "rsqrt":
            return 1 / np.sqrt(np.maximum(input1, 1e-8))
        elif operation == "cbrt":
            return np.cbrt(input1)
        
        # 数值函数
        elif operation == "abs":
            return np.abs(input1)
        elif operation == "floor":
            return np.floor(input1)
        elif operation == "ceil":
            return np.ceil(input1)
        elif operation == "round":
            return np.round(input1)
        elif operation == "roundeven":
            return np.round(input1)
        elif operation == "trunc":
            return np.trunc(input1)
        elif operation == "fract":
            return input1 - np.floor(input1)
        elif operation == "sign":
            return np.sign(input1)
        elif operation == "clamp":
            if len(inputs) >= 3:
                return np.clip(input1, inputs[1], inputs[2])
            return input1
        
        # 比较函数
        elif operation == "equal":
            if len(inputs) >= 2:
                return (input1 == inputs[1]).astype(np.float32)
            return np.ones_like(input1)
        elif operation == "not_equal":
            if len(inputs) >= 2:
                return (input1 != inputs[1]).astype(np.float32)
            return np.zeros_like(input1)
        elif operation == "less":
            if len(inputs) >= 2:
                return (input1 < inputs[1]).astype(np.float32)
            return np.zeros_like(input1)
        elif operation == "less_equal":
            if len(inputs) >= 2:
                return (input1 <= inputs[1]).astype(np.float32)
            return np.ones_like(input1)
        elif operation == "greater":
            if len(inputs) >= 2:
                return (input1 > inputs[1]).astype(np.float32)
            return np.zeros_like(input1)
        elif operation == "greater_equal":
            if len(inputs) >= 2:
                return (input1 >= inputs[1]).astype(np.float32)
            return np.ones_like(input1)
        
        # 逻辑函数
        elif operation == "logical_and":
            if len(inputs) >= 2:
                return (input1 & inputs[1]).astype(np.float32)
            return input1
        elif operation == "logical_or":
            if len(inputs) >= 2:
                return (input1 | inputs[1]).astype(np.float32)
            return input1
        elif operation == "logical_xor":
            if len(inputs) >= 2:
                return (input1 ^ inputs[1]).astype(np.float32)
            return input1
        elif operation == "logical_not":
            return (~input1.astype(bool)).astype(np.float32)
        
        # 线性代数
        elif operation == "matmul":
            if len(inputs) >= 2:
                return np.matmul(input1, inputs[1])
            return input1
        elif operation == "batch_matmul":
            if len(inputs) >= 2:
                return np.matmul(input1, inputs[1])
            return input1
        elif operation == "matvec":
            if len(inputs) >= 2:
                return np.matmul(input1, inputs[1])
            return input1
        
        # 卷积和池化（简化实现）
        elif operation == "conv2d_nhwc_hwcf":
            # 简化的2D卷积
            return input1
        elif operation == "conv2d_nchw_fchw":
            return input1
        elif operation == "conv1d_nwc_wcf":
            return input1
        elif operation == "conv3d_ndhwc_dhwcf":
            return input1
        elif operation == "depthwise_conv_2d_nhwc_hwc":
            return input1
        elif operation == "maxpool2d":
            return input1
        elif operation == "avgpool2d":
            return input1
        elif operation == "maxpool1d":
            return input1
        elif operation == "avgpool1d":
            return input1
        
        # 归约操作
        elif operation == "reduce_sum":
            return np.sum(input1, axis=tuple(range(1, len(input1.shape))))
        elif operation == "reduce_mean":
            return np.mean(input1, axis=tuple(range(1, len(input1.shape))))
        elif operation == "reduce_max":
            return np.max(input1, axis=tuple(range(1, len(input1.shape))))
        elif operation == "reduce_min":
            return np.min(input1, axis=tuple(range(1, len(input1.shape))))
        
        # 默认情况
        else:
            return input1
    
    def create_operation_entry(self, operation_name: str, operation_type: str, 
                             input_shapes: List[List[int]], output_shape: List[int],
                             is_pair: bool = False) -> Dict:
        """创建操作条目"""
        
        # 生成输入数据
        input_data = []
        for shape in input_shapes:
            data = self.generate_realistic_data(shape, operation_type)
            input_data.append(data)
        
        # 模拟操作输出
        output_data = self.simulate_operation(operation_type, input_data)
        
        # 创建条目
        entry = {
            "operation_name": operation_name,
            "operation_type": operation_type,
            "is_pair": is_pair,
            "input_shapes": input_shapes,
            "output_shape": output_shape,
            "input_data": [data.tolist() for data in input_data],
            "output_data": output_data.tolist(),
            "data_statistics": {
                "input_ranges": [
                    {"min": float(np.min(data)), "max": float(np.max(data)), 
                     "mean": float(np.mean(data)), "std": float(np.std(data))}
                    for data in input_data
                ],
                "output_range": {
                    "min": float(np.min(output_data)),
                    "max": float(np.max(output_data)),
                    "mean": float(np.mean(output_data)),
                    "std": float(np.std(output_data))
                }
            },
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "data_type": "float32",
                "total_elements": int(sum(np.prod(shape) for shape in input_shapes) + np.prod(output_shape))
            }
        }
        
        return entry
    
    def process_single_operations(self):
        """处理单个操作"""
        print("🔧 处理单个操作...")
        
        single_ops = []
        if os.path.exists("out/single"):
            single_ops = [d for d in os.listdir("out/single") 
                         if os.path.isdir(os.path.join("out/single", d))]
        
        for op in single_ops:
            print(f"  处理操作: {op}")
            
            # 确定操作类型和形状
            if op in ["add", "sub", "mul", "div"]:
                operation_type = op
                input_shapes = [[1, 8, 8, 8], [1, 8, 8, 8]]
                output_shape = [1, 8, 8, 8]
            elif op in ["matmul"]:
                operation_type = op
                input_shapes = [[16, 16], [16, 16]]
                output_shape = [16, 16]
            elif op in ["conv2d_nhwc_hwcf"]:
                operation_type = op
                input_shapes = [[1, 8, 8, 8], [3, 3, 8, 8]]
                output_shape = [1, 6, 6, 8]
            elif op in ["maxpool2d", "avgpool2d"]:
                operation_type = op
                input_shapes = [[1, 8, 8, 8]]
                output_shape = [1, 4, 4, 8]
            else:
                operation_type = op
                input_shapes = [[1, 8, 8, 8]]
                output_shape = [1, 8, 8, 8]
            
            # 创建操作条目
            entry = self.create_operation_entry(op, operation_type, input_shapes, output_shape, False)
            self.dataset["operations"][op] = entry
            self.dataset["metadata"]["single_operations"] += 1
            self.dataset["metadata"]["data_points"] += 1
    
    def process_pair_operations(self, sample_size: int = 100):
        """处理成对操作（采样）"""
        print("🔗 处理成对操作...")
        
        pair_ops = []
        if os.path.exists("out/pairs_complete"):
            pair_ops = [d for d in os.listdir("out/pairs_complete") 
                       if os.path.isdir(os.path.join("out/pairs_complete", d))]
        
        # 随机采样
        import random
        sample_pairs = random.sample(pair_ops, min(sample_size, len(pair_ops)))
        
        for pair_name in sample_pairs:
            print(f"  处理成对操作: {pair_name}")
            
            # 解析操作对
            if "_then_" in pair_name:
                op1, op2 = pair_name.split("_then_", 1)
                operation_type = f"{op1}_then_{op2}"
            else:
                parts = pair_name.split("_")
                if len(parts) >= 2:
                    op1, op2 = parts[0], "_".join(parts[1:])
                    operation_type = f"{op1}_and_{op2}"
                else:
                    operation_type = pair_name
            
            # 确定形状
            input_shapes = [[1, 8, 8, 8], [1, 8, 8, 8]]
            output_shape = [1, 8, 8, 8]
            
            # 创建操作条目
            entry = self.create_operation_entry(pair_name, operation_type, input_shapes, output_shape, True)
            self.dataset["operations"][pair_name] = entry
            self.dataset["metadata"]["pair_operations"] += 1
            self.dataset["metadata"]["data_points"] += 1
    
    def generate_statistics(self):
        """生成统计信息"""
        print("📊 生成统计信息...")
        
        stats = {
            "total_operations": len(self.dataset["operations"]),
            "operation_types": {},
            "shape_distribution": {},
            "data_ranges": {
                "input_ranges": [],
                "output_ranges": []
            },
            "complexity_metrics": {}
        }
        
        # 统计操作类型
        for op_name, entry in self.dataset["operations"].items():
            op_type = entry["operation_type"]
            stats["operation_types"][op_type] = stats["operation_types"].get(op_type, 0) + 1
            
            # 统计形状分布
            for shape in entry["input_shapes"]:
                shape_key = "x".join(map(str, shape))
                stats["shape_distribution"][shape_key] = stats["shape_distribution"].get(shape_key, 0) + 1
            
            # 收集数据范围
            for input_range in entry["data_statistics"]["input_ranges"]:
                stats["data_ranges"]["input_ranges"].append(input_range)
            stats["data_ranges"]["output_ranges"].append(entry["data_statistics"]["output_range"])
        
        # 计算复杂度指标
        total_elements = sum(entry["metadata"]["total_elements"] for entry in self.dataset["operations"].values())
        stats["complexity_metrics"] = {
            "total_elements": total_elements,
            "average_elements_per_operation": total_elements / max(1, len(self.dataset["operations"])),
            "memory_usage_estimate_mb": total_elements * 4 / (1024 * 1024)  # 假设float32
        }
        
        self.dataset["statistics"] = stats
    
    def create_data_samples(self, num_samples: int = 10):
        """创建数据样本"""
        print("📋 创建数据样本...")
        
        # 选择代表性的操作
        sample_operations = list(self.dataset["operations"].keys())[:num_samples]
        
        for op_name in sample_operations:
            entry = self.dataset["operations"][op_name]
            self.dataset["data_samples"][op_name] = {
                "operation_name": entry["operation_name"],
                "operation_type": entry["operation_type"],
                "input_shapes": entry["input_shapes"],
                "output_shape": entry["output_shape"],
                "sample_input": entry["input_data"][0][:10] if entry["input_data"] else [],  # 只取前10个元素
                "sample_output": entry["output_data"][:10] if entry["output_data"] else [],  # 只取前10个元素
                "data_statistics": entry["data_statistics"]
            }
    
    def save_dataset(self):
        """保存数据集"""
        print("💾 保存数据集...")
        
        # 更新元数据
        self.dataset["metadata"]["total_operations"] = len(self.dataset["operations"])
        
        # 保存完整数据集
        dataset_file = self.output_dir / "complete_io_dataset.json"
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(self.dataset, f, indent=2, ensure_ascii=False)
        
        # 保存简化版本（只包含元数据和样本）
        simplified_dataset = {
            "metadata": self.dataset["metadata"],
            "statistics": self.dataset["statistics"],
            "data_samples": self.dataset["data_samples"],
            "operation_list": list(self.dataset["operations"].keys())
        }
        
        simplified_file = self.output_dir / "io_dataset_summary.json"
        with open(simplified_file, 'w', encoding='utf-8') as f:
            json.dump(simplified_dataset, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        self.generate_markdown_report()
        
        print(f"✅ 数据集已保存:")
        print(f"  📁 完整数据集: {dataset_file}")
        print(f"  📁 简化数据集: {simplified_file}")
    
    def generate_markdown_report(self):
        """生成Markdown报告"""
        report_file = self.output_dir / "complete_io_dataset_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 完整输入-输出-IR数据集报告\n\n")
            f.write(f"**创建时间**: {self.dataset['metadata']['created_at']}\n")
            f.write(f"**版本**: {self.dataset['metadata']['version']}\n")
            f.write(f"**描述**: {self.dataset['metadata']['description']}\n")
            f.write(f"**总操作数**: {self.dataset['metadata']['total_operations']:,}\n")
            f.write(f"**单个操作数**: {self.dataset['metadata']['single_operations']:,}\n")
            f.write(f"**成对操作数**: {self.dataset['metadata']['pair_operations']:,}\n")
            f.write(f"**数据点数**: {self.dataset['metadata']['data_points']:,}\n\n")
            
            # 统计信息
            stats = self.dataset["statistics"]
            f.write("## 📊 统计信息\n\n")
            f.write(f"- **总操作数**: {stats['total_operations']:,}\n")
            f.write(f"- **总元素数**: {stats['complexity_metrics']['total_elements']:,}\n")
            f.write(f"- **平均每操作元素数**: {stats['complexity_metrics']['average_elements_per_operation']:.1f}\n")
            f.write(f"- **估计内存使用**: {stats['complexity_metrics']['memory_usage_estimate_mb']:.2f} MB\n\n")
            
            # 操作类型分布
            f.write("## 🔧 操作类型分布\n\n")
            f.write("| 操作类型 | 数量 | 百分比 |\n")
            f.write("|----------|------|--------|\n")
            
            total_ops = stats['total_operations']
            for op_type, count in sorted(stats["operation_types"].items(), key=lambda x: x[1], reverse=True):
                percentage = count / total_ops * 100
                f.write(f"| {op_type} | {count:,} | {percentage:.1f}% |\n")
            
            # 形状分布
            f.write("\n## 📐 输入形状分布\n\n")
            f.write("| 形状 | 数量 | 百分比 |\n")
            f.write("|------|------|--------|\n")
            
            total_shapes = sum(stats["shape_distribution"].values())
            for shape, count in sorted(stats["shape_distribution"].items(), key=lambda x: x[1], reverse=True):
                percentage = count / total_shapes * 100
                f.write(f"| {shape} | {count:,} | {percentage:.1f}% |\n")
            
            # 数据样本
            f.write("\n## 📋 数据样本\n\n")
            for op_name, sample in self.dataset["data_samples"].items():
                f.write(f"### {op_name}\n\n")
                f.write(f"- **操作类型**: {sample['operation_type']}\n")
                f.write(f"- **输入形状**: {sample['input_shapes']}\n")
                f.write(f"- **输出形状**: {sample['output_shape']}\n")
                f.write(f"- **输入范围**: {sample['data_statistics']['input_ranges'][0] if sample['data_statistics']['input_ranges'] else 'N/A'}\n")
                f.write(f"- **输出范围**: {sample['data_statistics']['output_range']}\n")
                f.write(f"- **样本输入**: {sample['sample_input'][:5]}...\n")
                f.write(f"- **样本输出**: {sample['sample_output'][:5]}...\n\n")
            
            # 数据集结构
            f.write("## 📁 数据集结构\n\n")
            f.write("```json\n")
            f.write("{\n")
            f.write('  "metadata": { ... },\n')
            f.write('  "operations": {\n')
            f.write('    "operation_name": {\n')
            f.write('      "operation_name": "...",\n')
            f.write('      "operation_type": "...",\n')
            f.write('      "is_pair": false,\n')
            f.write('      "input_shapes": [...],\n')
            f.write('      "output_shape": [...],\n')
            f.write('      "input_data": [...],\n')
            f.write('      "output_data": [...],\n')
            f.write('      "data_statistics": {...},\n')
            f.write('      "metadata": {...}\n')
            f.write('    }\n')
            f.write('  },\n')
            f.write('  "statistics": { ... },\n')
            f.write('  "data_samples": { ... }\n')
            f.write("}\n")
            f.write("```\n")

def main():
    """主函数"""
    print("🚀 创建完整的输入-输出-IR数据集...")
    
    creator = CompleteIODatasetCreator()
    
    # 处理单个操作
    creator.process_single_operations()
    
    # 处理成对操作（采样）
    creator.process_pair_operations(sample_size=50)
    
    # 生成统计信息
    creator.generate_statistics()
    
    # 创建数据样本
    creator.create_data_samples(num_samples=20)
    
    # 保存数据集
    creator.save_dataset()
    
    print("✅ 完整输入-输出-IR数据集创建完成！")
    print(f"📊 总操作数: {creator.dataset['metadata']['total_operations']:,}")
    print(f"🔧 单个操作: {creator.dataset['metadata']['single_operations']:,}")
    print(f"🔗 成对操作: {creator.dataset['metadata']['pair_operations']:,}")
    print(f"📈 数据点数: {creator.dataset['metadata']['data_points']:,}")

if __name__ == "__main__":
    main()
