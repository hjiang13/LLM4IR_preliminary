#!/usr/bin/env python3
"""
MLIR操作枚举脚本
从YAML配置生成具体的操作实例和所有可能的成对操作组合
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
    with open(config_dir / "ops_complete_v2.yaml", 'r') as f:
        ops_data = yaml.safe_load(f)
        config['ops'] = ops_data['operations']
        print(f"调试: 加载了 {len(config['ops'])} 个操作")
        print(f"调试: 第一个操作: {config['ops'][0]}")
    
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
    params['N_matmul'] = shape_config.get('matrix_dims', [16])
    
    return params

def generate_single_cases(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """生成单个操作的实例"""
    cases = []
    shape_params = expand_shape_params(config['shapes'])
    
    for op in config['ops']:
        op_id = op['id']
        
        # 根据操作类型选择参数组合
        if op_id == 'matmul':
            # 矩阵乘法：M, K, N
            param_combinations = itertools.product(
                shape_params['M'], 
                shape_params['K'], 
                shape_params['N_matmul']
            )
            for M, K, N in param_combinations:
                case = {
                    'case_id': f"{op_id}_M{M}_K{K}_N{N}",
                    'op_id': op_id,
                    'template': 'generic',
                    'params': {'M': M, 'K': K, 'N': N, 'dtype': 'f32'},
                    'inputs': [
                        {'name': 'A', 'shape': [M, K], 'dtype': 'f32'},
                        {'name': 'B', 'shape': [K, N], 'dtype': 'f32'}
                    ],
                    'outputs': [{'shape': [M, N], 'dtype': 'f32'}]
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
                    'template': 'generic',
                    'params': {'N': N, 'H': H, 'W': W, 'C': C, 'dtype': 'f32'},
                    'inputs': [{'name': 'x', 'shape': [N, H, W, C], 'dtype': 'f32'}],
                    'outputs': [{'shape': [N, H, W, C], 'dtype': 'f32'}]
                }
                
                # 特殊处理二元操作
                if op_id in ['add', 'mul', 'sub', 'div']:
                    case['inputs'].append({'name': 'y', 'shape': [N, H, W, C], 'dtype': 'f32'})
                
                cases.append(case)
    
    return cases

def generate_pair_cases(single_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """生成所有可能的成对操作组合"""
    pair_cases = []
    
    print(f"🔧 生成成对操作组合...")
    print(f"  单个操作数量: {len(single_cases)}")
    print(f"  理论组合总数: {len(single_cases) * len(single_cases)}")
    
    # 生成所有可能的组合（包括自组合）
    for i, op_a in enumerate(single_cases):
        for j, op_b in enumerate(single_cases):
            # 创建成对操作实例
            pair_case = {
                'case_id': f"{op_a['case_id']}_then_{op_b['case_id']}",
                'op_a': op_a,
                'op_b': op_b,
                'shape': op_a['inputs'][0]['shape'],  # 使用第一个操作的输入形状
                'dtype': op_a['inputs'][0]['dtype']
            }
            pair_cases.append(pair_case)
            
            # 每生成1000个组合显示进度
            if len(pair_cases) % 1000 == 0:
                print(f"    已生成: {len(pair_cases)} 个组合")
    
    return pair_cases

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
    print("🔧 生成成对操作实例...")
    pair_cases = generate_pair_cases(single_cases)
    
    # 保存结果
    print("💾 保存结果...")
    with open(out_dir / "cases_single_complete.json", 'w') as f:
        json.dump(single_cases, f, indent=2, ensure_ascii=False)
    
    with open(out_dir / "cases_pairs_complete.json", 'w') as f:
        json.dump(pair_cases, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 完成！")
    print(f"  单个操作实例: {len(single_cases)}")
    print(f"  成对操作实例: {len(pair_cases)}")
    print(f"  理论组合总数: {len(single_cases) * len(single_cases)}")
    print(f"  结果保存到: {out_dir}")

if __name__ == "__main__":
    main()
