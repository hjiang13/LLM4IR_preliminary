#!/usr/bin/env python3
"""
单个操作MLIR生成脚本
生成单个操作的MLIR文件
"""

import json
import os
import yaml
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from typing import Dict, Any, List

def load_cases(cases_file: Path) -> List[Dict[str, Any]]:
    """加载操作实例"""
    with open(cases_file, 'r') as f:
        return json.load(f)

def load_ops_config(config_file: Path) -> Dict[str, Any]:
    """加载操作配置"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def setup_jinja_env(template_dir: Path) -> Environment:
    """设置Jinja2环境"""
    env = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True
    )
    return env

def generate_simple_elementwise_mlir(case: Dict[str, Any]) -> str:
    """生成简单的元素级操作MLIR"""
    params = case['params']
    op_id = case['op_id']
    
    # 基本形状参数
    N, H, W, C = params['N'], params['H'], params['W'], params['C']
    dtype = params['dtype']
    
    # 根据操作类型生成不同的MLIR内容
    if op_id == 'relu':
        return f"""module {{
  func.func @main(%x: tensor<{N}x{H}x{W}x{C}x{dtype}>) -> tensor<{N}x{H}x{W}x{C}x{dtype}> {{
    %cst0 = arith.constant 0.0 : {dtype}
    %init = tensor.empty() : tensor<{N}x{H}x{W}x{C}x{dtype}>
    
    %result = linalg.generic {{
      indexing_maps = [
        affine_map<(i, j, k, l) -> (i, j, k, l)>,
        affine_map<(i, j, k, l) -> (i, j, k, l)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    }} ins(%x : tensor<{N}x{H}x{W}x{C}x{dtype}>) 
        outs(%init : tensor<{N}x{H}x{W}x{C}x{dtype}>) {{
      ^bb0(%x_val: {dtype}, %out: {dtype}):
        %max = arith.maximumf %x_val, %cst0 : {dtype}
        linalg.yield %max : {dtype}
    }} -> tensor<{N}x{H}x{W}x{C}x{dtype}>
    
    return %result : tensor<{N}x{H}x{W}x{C}x{dtype}>
  }}
}}"""
    
    elif op_id == 'exp':
        return f"""module {{
  func.func @main(%x: tensor<{N}x{H}x{W}x{C}x{dtype}>) -> tensor<{N}x{H}x{W}x{C}x{dtype}> {{
    %init = tensor.empty() : tensor<{N}x{H}x{W}x{C}x{dtype}>
    
    %result = linalg.generic {{
      indexing_maps = [
        affine_map<(i, j, k, l) -> (i, j, k, l)>,
        affine_map<(i, j, k, l) -> (i, j, k, l)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    }} ins(%x : tensor<{N}x{H}x{W}x{C}x{dtype}>) 
        outs(%init : tensor<{N}x{H}x{W}x{C}x{dtype}>) {{
      ^bb0(%x_val: {dtype}, %out: {dtype}):
        %exp = math.exp %x_val : {dtype}
        linalg.yield %exp : {dtype}
    }} -> tensor<{N}x{H}x{W}x{C}x{dtype}>
    
    return %result : tensor<{N}x{H}x{W}x{C}x{dtype}>
  }}
}}"""
    
    elif op_id == 'tanh':
        return f"""module {{
  func.func @main(%x: tensor<{N}x{H}x{W}x{C}x{dtype}>) -> tensor<{N}x{H}x{W}x{C}x{dtype}> {{
    %init = tensor.empty() : tensor<{N}x{H}x{W}x{C}x{dtype}>
    
    %result = linalg.generic {{
      indexing_maps = [
        affine_map<(i, j, k, l) -> (i, j, k, l)>,
        affine_map<(i, j, k, l) -> (i, j, k, l)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    }} ins(%x : tensor<{N}x{H}x{W}x{C}x{dtype}>) 
        outs(%init : tensor<{N}x{H}x{W}x{C}x{dtype}>) {{
      ^bb0(%x_val: {dtype}, %out: {dtype}):
        %tanh = math.tanh %x_val : {dtype}
        linalg.yield %tanh : {dtype}
    }} -> tensor<{N}x{H}x{W}x{C}x{dtype}>
    
    return %result : tensor<{N}x{H}x{W}x{C}x{dtype}>
  }}
}}"""
    
    elif op_id == 'sigmoid':
        return f"""module {{
  func.func @main(%x: tensor<{N}x{H}x{W}x{C}x{dtype}>) -> tensor<{N}x{H}x{W}x{C}x{dtype}> {{
    %cst_neg1 = arith.constant -1.0 : {dtype}
    %cst1 = arith.constant 1.0 : {dtype}
    %init = tensor.empty() : tensor<{N}x{H}x{W}x{C}x{dtype}>
    
    %result = linalg.generic {{
      indexing_maps = [
        affine_map<(i, j, k, l) -> (i, j, k, l)>,
        affine_map<(i, j, k, l) -> (i, j, k, l)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    }} ins(%x : tensor<{N}x{H}x{W}x{C}x{dtype}>) 
        outs(%init : tensor<{N}x{H}x{W}x{C}x{dtype}>) {{
      ^bb0(%x_val: {dtype}, %out: {dtype}):
        %neg_x = arith.mulf %x_val, %cst_neg1 : {dtype}
        %exp_neg_x = math.exp %neg_x : {dtype}
        %denom = arith.addf %cst1, %exp_neg_x : {dtype}
        %sigmoid = arith.divf %cst1, %denom : {dtype}
        linalg.yield %sigmoid : {dtype}
    }} -> tensor<{N}x{H}x{W}x{C}x{dtype}>
    
    return %result : tensor<{N}x{H}x{W}x{C}x{dtype}>
  }}
}}"""
    
    elif op_id in ['add', 'sub', 'mul', 'div']:
        # 二元操作
        if op_id == 'add':
            op = 'addf'
        elif op_id == 'sub':
            op = 'subf'
        elif op_id == 'mul':
            op = 'mulf'
        elif op_id == 'div':
            op = 'divf'
        
        return f"""module {{
  func.func @main(%x: tensor<{N}x{H}x{W}x{C}x{dtype}>, %y: tensor<{N}x{H}x{W}x{C}x{dtype}>) -> tensor<{N}x{H}x{W}x{C}x{dtype}> {{
    %init = tensor.empty() : tensor<{N}x{H}x{W}x{C}x{dtype}>
    
    %result = linalg.generic {{
      indexing_maps = [
        affine_map<(i, j, k, l) -> (i, j, k, l)>,
        affine_map<(i, j, k, l) -> (i, j, k, l)>,
        affine_map<(i, j, k, l) -> (i, j, k, l)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    }} ins(%x, %y : tensor<{N}x{H}x{W}x{C}x{dtype}>, tensor<{N}x{H}x{W}x{C}x{dtype}>) 
        outs(%init : tensor<{N}x{H}x{W}x{C}x{dtype}>) {{
      ^bb0(%x_val: {dtype}, %y_val: {dtype}, %out: {dtype}):
        %res = arith.{op} %x_val, %y_val : {dtype}
        linalg.yield %res : {dtype}
    }} -> tensor<{N}x{H}x{W}x{C}x{dtype}>
    
    return %result : tensor<{N}x{H}x{W}x{C}x{dtype}>
  }}
}}"""
    
    else:
        # 默认操作 - 简单的加法
        return f"""module {{
  func.func @main(%x: tensor<{N}x{H}x{W}x{C}x{dtype}>) -> tensor<{N}x{H}x{W}x{C}x{dtype}> {{
    %init = tensor.empty() : tensor<{N}x{H}x{W}x{C}x{dtype}>
    
    %result = linalg.generic {{
      indexing_maps = [
        affine_map<(i, j, k, l) -> (i, j, k, l)>,
        affine_map<(i, j, k, l) -> (i, j, k, l)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    }} ins(%x : tensor<{N}x{H}x{W}x{C}x{dtype}>) 
        outs(%init : tensor<{N}x{H}x{W}x{C}x{dtype}>) {{
      ^bb0(%x_val: {dtype}, %out: {dtype}):
        %res = arith.addf %x_val, %x_val : {dtype}
        linalg.yield %res : {dtype}
    }} -> tensor<{N}x{H}x{W}x{C}x{dtype}>
    
    return %result : tensor<{N}x{H}x{W}x{C}x{dtype}>
  }}
}}"""

def generate_matmul_mlir(case: Dict[str, Any]) -> str:
    """生成矩阵乘法的MLIR"""
    params = case['params']
    M, K, N = params['M'], params['K'], params['N']
    dtype = params['dtype']
    
    return f"""module {{
  func.func @main(
    %A: tensor<{M}x{K}x{dtype}>,
    %B: tensor<{K}x{N}x{dtype}>
  ) -> tensor<{M}x{N}x{dtype}> {{
    %init = tensor.empty() : tensor<{M}x{N}x{dtype}>
    
    %C = linalg.matmul ins(%A, %B : tensor<{M}x{K}x{dtype}>, tensor<{K}x{N}x{dtype}>)
                      outs(%init : tensor<{M}x{N}x{dtype}>) -> tensor<{M}x{N}x{dtype}>
    
    return %C : tensor<{M}x{N}x{dtype}>
  }}
}}"""

def generate_conv_mlir(case: Dict[str, Any]) -> str:
    """生成卷积的MLIR"""
    params = case['params']
    N, H, W, C = params['N'], params['H'], params['W'], params['C']
    dtype = params['dtype']
    
    # 如果没有KH, KW, F参数，使用默认值
    KH = params.get('KH', 3)
    KW = params.get('KW', 3)
    F = params.get('F', C)
    OH = params.get('OH', H - KH + 1)
    OW = params.get('OW', W - KW + 1)
    
    return f"""module {{
  func.func @main(
    %input: tensor<{N}x{H}x{W}x{C}x{dtype}>,
    %kernel: tensor<{KH}x{KW}x{C}x{F}x{dtype}>
  ) -> tensor<{N}x{OH}x{OW}x{F}x{dtype}> {{
    %init = tensor.empty() : tensor<{N}x{OH}x{OW}x{F}x{dtype}>
    
    %output = linalg.conv_2d_nhwc_hwcf
      ins(%input, %kernel : tensor<{N}x{H}x{W}x{C}x{dtype}>, tensor<{KH}x{KW}x{C}x{F}x{dtype}>)
      outs(%init : tensor<{N}x{OH}x{OW}x{F}x{dtype}>) -> tensor<{N}x{OH}x{OW}x{F}x{dtype}>
    
    return %output : tensor<{N}x{OH}x{OW}x{F}x{dtype}>
  }}
}}"""

def generate_pooling_mlir(case: Dict[str, Any]) -> str:
    """生成池化的MLIR"""
    params = case['params']
    N, H, W, C = params['N'], params['H'], params['W'], params['C']
    dtype = params['dtype']
    
    # 如果没有OH, OW参数，使用默认值
    OH = params.get('OH', H // 2)
    OW = params.get('OW', W // 2)
    
    return f"""module {{
  func.func @main(
    %input: tensor<{N}x{H}x{W}x{C}x{dtype}>
  ) -> tensor<{N}x{OH}x{OW}x{C}x{dtype}> {{
    %init = tensor.empty() : tensor<{N}x{OH}x{OW}x{C}x{dtype}>
    
    %output = linalg.pooling_nhwc_max
      ins(%input : tensor<{N}x{H}x{W}x{C}x{dtype}>)
      outs(%init : tensor<{N}x{OH}x{OW}x{C}x{dtype}>)
      {{dilations = dense<1> : vector<2xi64>,
       strides = dense<2> : vector<2xi64>,
       windows = dense<2> : vector<2xi64>}}
      (%in: {dtype}, %out: {dtype}) {{
        %result = arith.maximumf %in, %out : {dtype}
        linalg.yield %result : {dtype}
      }} -> tensor<{N}x{OH}x{OW}x{C}x{dtype}>
    
    return %output : tensor<{N}x{OH}x{OW}x{C}x{dtype}>
  }}
}}"""

def generate_reduce_mlir(case: Dict[str, Any]) -> str:
    """生成归约的MLIR"""
    params = case['params']
    N, H, W, C = params['N'], params['H'], params['W'], params['C']
    dtype = params['dtype']
    
    return f"""module {{
  func.func @main(
    %input: tensor<{N}x{H}x{W}x{C}x{dtype}>
  ) -> tensor<{N}x{C}x{dtype}> {{
    %init = tensor.empty() : tensor<{N}x{C}x{dtype}>
    
    %output = linalg.reduce
      ins(%input : tensor<{N}x{H}x{W}x{C}x{dtype}>)
      outs(%init : tensor<{N}x{C}x{dtype}>)
      dimensions = [1, 2]
      (%in: {dtype}, %out: {dtype}) {{
        %result = arith.addf %in, %out : {dtype}
        linalg.yield %result : {dtype}
      }} -> tensor<{N}x{C}x{dtype}>
    
    return %output : tensor<{N}x{C}x{dtype}>
  }}
}}"""

def generate_mlir(case: Dict[str, Any]) -> str:
    """根据操作类型生成MLIR"""
    op_id = case['op_id']
    
    # 元素级操作
    elementwise_ops = [
        'relu', 'leaky_relu', 'elu', 'gelu', 'swish', 'mish',
        'exp', 'log', 'log2', 'log10', 'sqrt', 'rsqrt', 'cbrt',
        'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
        'sinh', 'cosh', 'asinh', 'acosh', 'atanh',
        'sigmoid', 'softplus', 'softsign', 'hard_sigmoid', 'hard_tanh'
    ]
    
    # 二元操作
    binary_ops = ['add', 'sub', 'mul', 'div', 'mod', 'pow', 'atan2']
    
    # 比较操作
    comparison_ops = ['equal', 'not_equal', 'greater', 'greater_equal', 'less', 'less_equal']
    
    # 逻辑操作
    logical_ops = ['logical_and', 'logical_or', 'logical_not', 'logical_xor']
    
    if op_id in elementwise_ops or op_id in binary_ops or op_id in comparison_ops or op_id in logical_ops:
        return generate_simple_elementwise_mlir(case)
    elif op_id == 'matmul':
        return generate_matmul_mlir(case)
    elif op_id == 'conv2d_nhwc_hwcf':
        return generate_conv_mlir(case)
    elif op_id in ['avgpool2d', 'maxpool2d']:
        return generate_pooling_mlir(case)
    elif op_id == 'reduce_sum':
        return generate_reduce_mlir(case)
    elif op_id == 'clamp':
        return generate_simple_elementwise_mlir(case)
    else:
        # 默认使用元素级操作
        return generate_simple_elementwise_mlir(case)

def main():
    """主函数"""
    # 设置路径
    cases_file = Path("out/cases_single.json")
    config_file = Path("config/ops.yaml")
    template_dir = Path("templates")
    output_dir = Path("out/single")
    
    # 检查输入文件
    if not cases_file.exists():
        print(f"❌ 找不到实例文件: {cases_file}")
        print("请先运行 scripts/enumerate_ops.py")
        return
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("📋 加载操作实例...")
    cases = load_cases(cases_file)
    
    print("📋 加载操作配置...")
    ops_config = load_ops_config(config_file)
    
    # 生成MLIR文件
    print("🔧 生成MLIR文件...")
    success_count = 0
    error_count = 0
    
    for case in cases:
        try:
            # 创建操作目录
            op_dir = output_dir / case['op_id']
            op_dir.mkdir(exist_ok=True)
            
            # 生成MLIR内容
            mlir_content = generate_mlir(case)
            
            # 保存文件
            output_file = op_dir / f"{case['case_id']}.mlir"
            with open(output_file, 'w') as f:
                f.write(mlir_content)
            
            success_count += 1
            print(f"✅ 成功生成: {case['case_id']}")
            
        except Exception as e:
            print(f"❌ 生成失败 {case['case_id']}: {e}")
            error_count += 1
    
    print(f"\n✅ 完成！")
    print(f"  成功: {success_count}")
    print(f"  失败: {error_count}")
    print(f"  输出目录: {output_dir}")

if __name__ == "__main__":
    main()
