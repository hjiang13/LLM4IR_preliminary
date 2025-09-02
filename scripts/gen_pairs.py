#!/usr/bin/env python3
"""
成对操作MLIR生成脚本
生成两个操作组合的MLIR文件
"""

import json
import os
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from typing import Dict, Any, List

def load_pair_cases(cases_file: str) -> List[Dict[str, Any]]:
    """加载成对操作实例"""
    with open(cases_file, 'r') as f:
        return json.load(f)

def generate_pair_mlir(pair_case: Dict[str, Any], env: Environment, out_dir: Path) -> bool:
    """生成单个成对操作的MLIR文件"""
    try:
        op_a = pair_case['op_a']
        op_b = pair_case['op_b']
        case_id = pair_case['case_id']
        
        # 创建输出目录
        op_dir = out_dir / "pairs" / case_id
        op_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成MLIR文件
        mlir_file = op_dir / f"{case_id}.mlir"
        
        # 根据操作类型选择模板
        if op_a['op_id'] in ['relu', 'add', 'mul', 'sub', 'div', 'exp', 'log', 'tanh', 'sigmoid', 'clamp']:
            template = env.get_template("linalg/elementwise_generic.mlir.j2")
        elif op_a['op_id'] == 'matmul':
            template = env.get_template("linalg/matmul_4d.mlir.j2")
        else:
            print(f"❌ 不支持的操作类型: {op_a['op_id']}")
            return False
        
        # 渲染模板
        params = op_a['params'].copy()
        if 'y' in op_a['inputs'][0]['name']:
            params['y'] = True
        
        # 设置表达式实现
        if op_a['op_id'] == 'relu':
            params['expr_impl'] = 'arith.maximumf %a, %cst : f32'
        elif op_a['op_id'] == 'add':
            params['expr_impl'] = 'arith.addf %a, %b : f32'
        elif op_a['op_id'] == 'mul':
            params['expr_impl'] = 'arith.mulf %a, %b : f32'
        elif op_a['op_id'] == 'sub':
            params['expr_impl'] = 'arith.subf %a, %b : f32'
        elif op_a['op_id'] == 'div':
            params['expr_impl'] = 'arith.divf %a, %b : f32'
        elif op_a['op_id'] == 'exp':
            params['expr_impl'] = 'math.exp %a : f32'
        elif op_a['op_id'] == 'log':
            params['expr_impl'] = 'math.log %a : f32'
        elif op_a['op_id'] == 'tanh':
            params['expr_impl'] = 'math.tanh %a : f32'
        elif op_a['op_id'] == 'sigmoid':
            params['expr_impl'] = 'arith.divf %cst, %sum : f32'
        elif op_a['op_id'] == 'clamp':
            params['expr_impl'] = 'arith.maximumf %min, %max : f32'
        elif op_a['op_id'] == 'matmul':
            params['expr_impl'] = 'arith.mulf %a, %b : f32'
        
        # 渲染模板
        mlir_content = template.render(**params)
        
        # 写入文件
        with open(mlir_file, 'w') as f:
            f.write(mlir_content)
        
        print(f"✅ 生成成功: {case_id}")
        return True
        
    except Exception as e:
        print(f"❌ 生成失败 {case_id}: {e}")
        return False

def main():
    """主函数"""
    print("🔗 生成成对操作MLIR文件...")
    
    # 设置Jinja2环境
    env = Environment(
        loader=FileSystemLoader('templates'),
        trim_blocks=True,
        lstrip_blocks=True
    )
    
    # 加载成对操作实例
    cases_file = "out/cases_pairs.json"
    if not os.path.exists(cases_file):
        print(f"❌ 成对操作实例文件不存在: {cases_file}")
        return
    
    pair_cases = load_pair_cases(cases_file)
    print(f"📋 加载了 {len(pair_cases)} 个成对操作实例")
    
    # 创建输出目录
    out_dir = Path("out")
    pairs_dir = out_dir / "pairs"
    pairs_dir.mkdir(exist_ok=True)
    
    # 生成MLIR文件
    success_count = 0
    total_count = len(pair_cases)
    
    for pair_case in pair_cases:
        if generate_pair_mlir(pair_case, env, out_dir):
            success_count += 1
    
    print(f"🎉 完成！")
    print(f"  成功: {success_count}/{total_count}")
    print(f"  成功率: {success_count/total_count*100:.1f}%")

if __name__ == "__main__":
    main()
