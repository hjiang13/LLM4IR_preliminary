#!/usr/bin/env python3

import json
from pathlib import Path

def load_config(config_dir: Path) -> dict:
    """加载配置文件"""
    with open(config_dir / "ops.yaml", 'r') as f:
        ops = yaml.safe_load(f)
    
    with open(config_dir / "dtypes.yaml", 'r') as f:
        dtypes = yaml.safe_load(f)
    
    with open(config_dir / "shapes.yaml", 'r') as f:
        shapes = yaml.safe_load(f)
    
    return {
        'ops': ops,
        'dtypes': dtypes,
        'shapes': shapes
    }

def test_pair_generation():
    """测试成对操作生成"""
    try:
        # 加载单个操作实例
        with open("out/cases_single.json", 'r') as f:
            single_cases = json.load(f)
        
        print(f"单个操作实例数量: {len(single_cases)}")
        
        # 测试成对操作生成
        pairs = []
        for i, case_a in enumerate(single_cases):
            for j, case_b in enumerate(single_cases):
                if i == j:
                    continue
                
                # 检查形状兼容性
                if is_shape_compatible(case_a, case_b):
                    pair_case = {
                        'case_id': f"{case_a['case_id']}_then_{case_b['case_id']}",
                        'op_a': case_a,
                        'op_b': case_b,
                        'status': 'compatible'
                    }
                    pairs.append(pair_case)
        
        print(f"成对操作实例数量: {len(pairs)}")
        
        # 保存成对操作实例
        with open("out/cases_pairs.json", 'w') as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)
        
        print("✅ 成对操作实例生成成功！")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

def is_shape_compatible(case_a: dict, case_b: dict) -> bool:
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

if __name__ == "__main__":
    test_pair_generation()
