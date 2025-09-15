#!/usr/bin/env python3
"""
LLM4IR实验配置
"""

import json
import os
from pathlib import Path

class ExperimentConfig:
    """实验配置类"""
    
    def __init__(self):
        self.base_dir = Path(".")
        self.output_dir = self.base_dir / "experiments"
        self.log_dir = self.output_dir / "logs"
        self.results_dir = self.output_dir / "results"
        
        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # 实验参数
        self.config = {
            "experiment_name": "LLM4IR_Preliminary",
            "version": "1.0",
            "description": "LLM4IR初步实验 - 基于43,006个IR文件",
            
            # 测试参数
            "test_parameters": {
                "single_operations": {
                    "enabled": True,
                    "sample_size": 10,  # 随机采样数量
                    "operations": ["add", "relu", "matmul", "conv2d_nhwc_hwcf"]
                },
                "pair_operations": {
                    "enabled": True,
                    "sample_size": 5,
                    "combinations": ["relu_gelu", "add_sigmoid", "matmul_tanh"]
                },
                "performance_test": {
                    "enabled": True,
                    "iterations": 100,
                    "warmup_iterations": 10
                }
            },
            
            # 优化传递配置
            "optimization_passes": {
                "basic": [
                    "convert-linalg-to-loops",
                    "lower-affine", 
                    "convert-scf-to-cf"
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
                    "arith-bufferize",
                    "tensor-bufferize", 
                    "finalizing-bufferize",
                    "convert-linalg-to-loops",
                    "lower-affine",
                    "convert-scf-to-cf",
                    "convert-math-to-llvm",
                    "convert-arith-to-llvm",
                    "convert-func-to-llvm",
                    "memref-expand",
                    "convert-memref-to-llvm",
                    "reconcile-unrealized-casts"
                ]
            },
            
            # 输入输出配置
            "tensor_configs": {
                "small": "tensor<1x8x8x8xf32>",
                "medium": "tensor<16x16xf32>", 
                "large": "tensor<32x32xf32>",
                "conv_input": "tensor<1x8x8x8xf32>",
                "conv_kernel": "tensor<3x3x8x8xf32>",
                "conv_output": "tensor<1x6x6x8xf32>"
            },
            
            # 性能指标
            "metrics": {
                "compilation_time": True,
                "execution_time": True,
                "memory_usage": True,
                "code_size": True,
                "optimization_effectiveness": True
            }
        }
    
    def save_config(self):
        """保存配置到文件"""
        config_file = self.output_dir / "experiment_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        print(f"✅ 实验配置已保存到: {config_file}")
    
    def load_catalog(self):
        """加载IR目录信息"""
        try:
            with open('ir_catalog_analysis.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("❌ 未找到目录分析文件")
            return None
    
    def get_operation_files(self, operation_name, file_type="mlir"):
        """获取操作对应的文件路径"""
        if file_type == "mlir":
            return f"out/single/{operation_name}/{operation_name}_N1_H8_W8_C8.mlir"
        elif file_type == "llvm":
            return f"out/single_llvm/{operation_name}/{operation_name}_N1_H8_W8_C8/{operation_name}_N1_H8_W8_C8.ll"
        else:
            return None

def main():
    """主函数"""
    print("🔧 设置LLM4IR实验环境...")
    
    config = ExperimentConfig()
    config.save_config()
    
    # 加载目录信息
    catalog = config.load_catalog()
    if catalog:
        print(f"📊 加载了 {catalog['summary']['total_files']:,} 个IR文件")
        print(f"🔧 单个操作: {catalog['summary']['single_operations']}")
        print(f"🔗 成对操作: {catalog['summary']['pair_operations']}")
    
    print("✅ 实验环境设置完成！")
    print(f"📁 输出目录: {config.output_dir}")
    print(f"📋 配置文件: {config.output_dir}/experiment_config.json")

if __name__ == "__main__":
    main()
