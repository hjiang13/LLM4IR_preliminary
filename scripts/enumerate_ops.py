#!/usr/bin/env python3
"""
MLIR操作枚举脚本
从YAML配置生成具体的操作实例
"""

import yaml
import json
import itertools
from pathlib import Path
from typing import Dict, List, Any

def load_config(config_dir: Path) -> Dict[str, Any]:
    """加载配置文件"""
    config = {}
    
    # 加载操作配置
    with open(config_dir / "ops.yaml", 'r') as f:
        config['ops'] = yaml.safe_load(f)
    
    # 加载数据类型配置
    with open(config_dir / "dtypes.yaml", 'r') as f:
        config['dtypes'] = yaml.safe_load(f)
    
    # 加载形状配置
    with open(config_dir / "shapes.yaml", 'r') as f:
        config['shapes'] = yaml.safe_load(f)
    
    return config

def expand_shape_params(shape_config: Dict[str, Any]) -> Dict[str, List[int]]:
    """展开形状参数网格"""
    params = {}
    
    # 批次大小
    params['N'] = shape_config.get('batch_sizes', [1])
    
    # 空间维度
    spatial = shape_config.get('spatial_dims', [8])
    params['H'] = spatial
    params['W'] = spatial
    
    # 通道数
    params['C'] = shape_config.get('channels', [8])
    
    # 矩阵维度
    params['M'] = shape_config.get('matrix_dims', [16])
    params['K'] = shape_config.get('matrix_dims', [16])
    params['N_matmul'] = shape_config.get('matrix_dims', [16])  # 避免与批次大小冲突
    
    # 卷积核大小
    params['KH'] = shape_config.get('kernel_sizes', [1, 3])
    params['KW'] = shape_config.get('kernel_sizes', [1, 3])
    
    # 输出特征数
    params['F'] = shape_config.get('output_features', [8])
    
    return params

def calculate_derived_shapes(op_config: Dict[str, Any], params: Dict[str, int]) -> Dict[str, int]:
    """计算派生形状（如卷积输出尺寸）"""
    derived = params.copy()
    
    if op_config['id'] == 'conv2d_nhwc_hwcf':
        # 卷积输出尺寸计算
        H, W = params['H'], params['W']
        KH, KW = params['KH'], params['KW']
        stride_h = op_config['attrs'].get('stride_h', 1)
        stride_w = op_config['attrs'].get('stride_w', 1)
        
        derived['OH'] = (H - KH) // stride_h + 1
        derived['OW'] = (W - KW) // stride_w + 1
        
    elif op_config['id'] in ['avgpool2d', 'maxpool2d']:
        # 池化输出尺寸计算
        H, W = params['H'], params['W']
        kernel_h = op_config['attrs'].get('kernel_h', 2)
        kernel_w = op_config['attrs'].get('kernel_w', 2)
        stride_h = op_config['attrs'].get('stride_h', 2)
        stride_w = op_config['attrs'].get('stride_w', 2)
        
        derived['OH'] = (H - kernel_h) // stride_h + 1
        derived['OW'] = (W - kernel_w) // stride_w + 1
    
    return derived

def generate_single_cases(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """生成单个操作的实例"""
    cases = []
    shape_params = expand_shape_params(config['shapes'])
    
    for op in config['ops']:
        op_id = op['id']
        
        # 根据操作类型选择参数组合
        if op_id == 'matmul':
            # 4D张量矩阵乘法：N, H, W, C
            param_combinations = itertools.product(
                shape_params['N'], 
                shape_params['H'], 
                shape_params['W'],
                shape_params['C']
            )
            for N, H, W, C in param_combinations:
                case = {
                    'case_id': f"{op_id}_N{N}_H{H}_W{W}_C{C}",
                    'op_id': op_id,
                    'template': op['template'],
                    'params': {'N': N, 'H': H, 'W': W, 'C': C, 'dtype': 'f32'},
                    'inputs': [
                        {'name': 'A', 'shape': [N, H, W, C], 'dtype': 'f32'},
                        {'name': 'B', 'shape': [N, H, W, C], 'dtype': 'f32'}
                    ],
                    'outputs': [{'shape': [N, H, W, C], 'dtype': 'f32'}]
                }
                cases.append(case)
                
        elif op_id == 'conv2d_nhwc_hwcf':
            # 卷积：N, H, W, C, KH, KW, F
            param_combinations = itertools.product(
                shape_params['N'], shape_params['H'], shape_params['W'],
                shape_params['C'], shape_params['KH'], shape_params['KW'],
                shape_params['F']
            )
            for N, H, W, C, KH, KW, F in param_combinations:
                # 计算输出尺寸
                OH = (H - KH) + 1  # 简化计算，假设stride=1, padding=VALID
                OW = (W - KW) + 1
                
                case = {
                    'case_id': f"{op_id}_N{N}_H{H}_W{W}_C{C}_KH{KH}_KW{KW}_F{F}",
                    'op_id': op_id,
                    'template': op['template'],
                    'params': {
                        'N': N, 'H': H, 'W': W, 'C': C,
                        'KH': KH, 'KW': KW, 'F': F,
                        'OH': OH, 'OW': OW, 'dtype': 'f32'
                    },
                    'inputs': [
                        {'name': 'I', 'shape': [N, H, W, C], 'dtype': 'f32'},
                        {'name': 'K', 'shape': [KH, KW, C, F], 'dtype': 'f32'}
                    ],
                    'outputs': [{'shape': [N, OH, OW, F], 'dtype': 'f32'}]
                }
                cases.append(case)
                
        elif op_id in ['avgpool2d', 'maxpool2d']:
            # 池化：N, H, W, C
            param_combinations = itertools.product(
                shape_params['N'], shape_params['H'], shape_params['W'],
                shape_params['C']
            )
            for N, H, W, C in param_combinations:
                # 计算输出尺寸
                OH = (H - 2) // 2 + 1  # 假设kernel=2, stride=2
                OW = (W - 2) // 2 + 1
                
                case = {
                    'case_id': f"{op_id}_N{N}_H{H}_W{W}_C{C}",
                    'op_id': op_id,
                    'template': op['template'],
                    'params': {
                        'N': N, 'H': H, 'W': W, 'C': C,
                        'OH': OH, 'OW': OW, 'dtype': 'f32',
                        'kernel_h': 2, 'kernel_w': 2,
                        'stride_h': 2, 'stride_w': 2
                    },
                    'inputs': [{'name': 'x', 'shape': [N, H, W, C], 'dtype': 'f32'}],
                    'outputs': [{'shape': [N, OH, OW, C], 'dtype': 'f32'}]
                }
                cases.append(case)
                
        elif op_id == 'reduce_sum':
            # 归约：N, H, W, C
            param_combinations = itertools.product(
                shape_params['N'], shape_params['H'], shape_params['W'],
                shape_params['C']
            )
            for N, H, W, C in param_combinations:
                case = {
                    'case_id': f"{op_id}_N{N}_H{H}_W{W}_C{C}",
                    'op_id': op_id,
                    'template': op['template'],
                    'params': {'N': N, 'H': H, 'W': W, 'C': C, 'dtype': 'f32'},
                    'inputs': [{'name': 'x', 'shape': [N, H, W, C], 'dtype': 'f32'}],
                    'outputs': [{'shape': [N, C], 'dtype': 'f32'}]
                }
                cases.append(case)
                
        else:
            # 元素级操作：N, H, W, C
            param_combinations = itertools.product(
                shape_params['N'], shape_params['H'], shape_params['W'],
                shape_params['C']
            )
            for N, H, W, C in param_combinations:
                case = {
                    'case_id': f"{op_id}_N{N}_H{H}_W{W}_C{C}",
                    'op_id': op_id,
                    'template': op['template'],
                    'params': {'N': N, 'H': H, 'W': W, 'C': C, 'dtype': 'f32'},
                    'inputs': [{'name': 'x', 'shape': [N, H, W, C], 'dtype': 'f32'}],
                    'outputs': [{'shape': [N, H, W, C], 'dtype': 'f32'}]
                }
                
                # 特殊处理二元操作
                if op_id in ['add', 'mul', 'sub', 'div']:
                    case['inputs'].append({'name': 'y', 'shape': [N, H, W, C], 'dtype': 'f32'})
                elif op_id == 'clamp':
                    case['inputs'].extend([
                        {'name': 'min_val', 'shape': [], 'dtype': 'f32'},
                        {'name': 'max_val', 'shape': [], 'dtype': 'f32'}
                    ])
                
                cases.append(case)
    
    return cases

def generate_pair_cases(single_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """生成成对操作的实例"""
    pairs = []
    max_pairs = 100  # 限制成对操作数量
    
    for i, case_a in enumerate(single_cases):
        for j, case_b in enumerate(single_cases):
            if i == j:
                continue  # 跳过自身组合
                
            # 检查形状兼容性
            if is_shape_compatible(case_a, case_b):
                pair_case = {
                    'case_id': f"{case_a['case_id']}_then_{case_b['case_id']}",
                    'op_a': case_a,
                    'op_b': case_b,
                    'status': 'compatible'
                }
                pairs.append(pair_case)
                
                # 限制成对操作数量
                if len(pairs) >= max_pairs:
                    break
        
        # 限制成对操作数量
        if len(pairs) >= max_pairs:
            break
    
    return pairs

def is_shape_compatible(case_a: Dict[str, Any], case_b: Dict[str, Any]) -> bool:
    """检查两个操作是否形状兼容"""
    # 简化检查：输出形状与输入形状匹配
    output_a = case_a['outputs'][0]
    input_b = case_b['inputs'][0]
    
    # 检查形状和数据类型
    if output_a['shape'] == input_b['shape'] and output_a['dtype'] == input_b['dtype']:
        return True
    
    # 检查广播兼容性（简化版本）
    if len(output_a['shape']) == len(input_b['shape']):
        compatible = True
        for dim_a, dim_b in zip(output_a['shape'], input_b['shape']):
            if dim_a != dim_b and dim_a != 1 and dim_b != 1:
                compatible = False
                break
        if compatible:
            return True
    
    return False

def main():
    """主函数"""
    config_dir = Path("config")
    out_dir = Path("out")
    
    # 创建输出目录
    out_dir.mkdir(exist_ok=True)
    
    # 加载配置
    print("📋 加载配置文件...")
    config = load_config(config_dir)
    
    # 生成单个操作实例
    print("🔧 生成单个操作实例...")
    single_cases = generate_single_cases(config)
    
    # 生成成对操作实例
    print("🔗 生成成对操作实例...")
    pair_cases = generate_pair_cases(single_cases)
    
    # 保存结果
    print("💾 保存结果...")
    with open(out_dir / "cases_single.json", 'w') as f:
        json.dump(single_cases, f, indent=2, ensure_ascii=False)
    
    with open(out_dir / "cases_pairs.json", 'w') as f:
        json.dump(pair_cases, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 完成！")
    print(f"  单个操作实例: {len(single_cases)}")
    print(f"  成对操作实例: {len(pair_cases)}")
    print(f"  结果保存到: {out_dir}")

if __name__ == "__main__":
    main()
