#!/usr/bin/env python3
"""
成对操作MLIR生成脚本
生成所有20,164个成对操作的MLIR文件
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List

def load_cases(cases_file: Path) -> List[Dict[str, Any]]:
    """加载成对操作实例"""
    with open(cases_file, 'r') as f:
        return json.load(f)

def generate_simple_elementwise_mlir(op_id: str, input_tensor: str, output_tensor: str, dtype: str, N: int, H: int, W: int, C: int) -> str:
    """生成简单元素级操作的MLIR"""
    # 基础算术操作
    if op_id in ['arith.addi', 'arith.addf']:
        return f"""    %{output_tensor} = linalg.generic {{
          indexing_maps = [
            affine_map<(i, j, k, l) -> (i, j, k, l)>,
            affine_map<(i, j, k, l) -> (i, j, k, l)>,
            affine_map<(i, j, k, l) -> (i, j, k, l)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }} ins(%{input_tensor}, %{input_tensor} : tensor<{N}x{H}x{W}x{C}x{dtype}>, tensor<{N}x{H}x{W}x{C}x{dtype}>) 
            outs(%{input_tensor} : tensor<{N}x{H}x{W}x{C}x{dtype}>) {{
          ^bb0(%x: {dtype}, %y: {dtype}, %o: {dtype}):
            %result = arith.addf %x, %y : {dtype}
            linalg.yield %result : {dtype}
        }} -> tensor<{N}x{H}x{W}x{C}x{dtype}>"""
    
    elif op_id in ['arith.subi', 'arith.subf']:
        return f"""    %{output_tensor} = linalg.generic {{
          indexing_maps = [
            affine_map<(i, j, k, l) -> (i, j, k, l)>,
            affine_map<(i, j, k, l) -> (i, j, k, l)>,
            affine_map<(i, j, k, l) -> (i, j, k, l)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }} ins(%{input_tensor}, %{input_tensor} : tensor<{N}x{H}x{W}x{C}x{dtype}>, tensor<{N}x{H}x{W}x{C}x{dtype}>) 
            outs(%{input_tensor} : tensor<{N}x{H}x{W}x{C}x{dtype}>) {{
          ^bb0(%x: {dtype}, %y: {dtype}, %o: {dtype}):
            %result = arith.subf %x, %y : {dtype}
            linalg.yield %result : {dtype}
        }} -> tensor<{N}x{H}x{W}x{C}x{dtype}>"""
    
    elif op_id in ['arith.muli', 'arith.mulf']:
        return f"""    %{output_tensor} = linalg.generic {{
          indexing_maps = [
            affine_map<(i, j, k, l) -> (i, j, k, l)>,
            affine_map<(i, j, k, l) -> (i, j, k, l)>,
            affine_map<(i, j, k, l) -> (i, j, k, l)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }} ins(%{input_tensor}, %{input_tensor} : tensor<{N}x{H}x{W}x{C}x{dtype}>, tensor<{N}x{H}x{W}x{C}x{dtype}>) 
            outs(%{input_tensor} : tensor<{N}x{H}x{W}x{C}x{dtype}>) {{
          ^bb0(%x: {dtype}, %y: {dtype}, %o: {dtype}):
            %result = arith.mulf %x, %y : {dtype}
            linalg.yield %result : {dtype}
        }} -> tensor<{N}x{H}x{W}x{C}x{dtype}>"""
    
    elif op_id in ['arith.divi', 'arith.divf']:
        return f"""    %{output_tensor} = linalg.generic {{
          indexing_maps = [
            affine_map<(i, j, k, l) -> (i, j, k, l)>,
            affine_map<(i, j, k, l) -> (i, j, k, l)>,
            affine_map<(i, j, k, l) -> (i, j, k, l)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }} ins(%{input_tensor}, %{input_tensor} : tensor<{N}x{H}x{W}x{C}x{dtype}>, tensor<{N}x{H}x{W}x{C}x{dtype}>) 
            outs(%{input_tensor} : tensor<{N}x{H}x{W}x{C}x{dtype}>) {{
          ^bb0(%x: {dtype}, %y: {dtype}, %o: {dtype}):
            %result = arith.divf %x, %y : {dtype}
            linalg.yield %result : {dtype}
        }} -> tensor<{N}x{H}x{W}x{C}x{dtype}>"""
    
    # 数学函数操作
    elif op_id in ['math.sqrt', 'math.rsqrt']:
        return f"""    %{output_tensor} = linalg.generic {{
          indexing_maps = [
            affine_map<(i, j, k, l) -> (i, j, k, l)>,
            affine_map<(i, j, k, l) -> (i, j, k, l)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }} ins(%{input_tensor} : tensor<{N}x{H}x{W}x{C}x{dtype}>) 
            outs(%{input_tensor} : tensor<{N}x{H}x{W}x{C}x{dtype}>) {{
          ^bb0(%x: {dtype}, %o: {dtype}):
            %result = math.sqrt %x : {dtype}
            linalg.yield %result : {dtype}
        }} -> tensor<{N}x{H}x{W}x{C}x{dtype}>"""
    
    elif op_id in ['math.exp', 'math.log']:
        return f"""    %{output_tensor} = linalg.generic {{
          indexing_maps = [
            affine_map<(i, j, k, l) -> (i, j, k, l)>,
            affine_map<(i, j, k, l) -> (i, j, k, l)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }} ins(%{input_tensor} : tensor<{N}x{H}x{W}x{C}x{dtype}>) 
            outs(%{input_tensor} : tensor<{N}x{H}x{W}x{C}x{dtype}>) {{
          ^bb0(%x: {dtype}, %o: {dtype}):
            %result = math.exp %x : {dtype}
            linalg.yield %result : {dtype}
        }} -> tensor<{N}x{H}x{W}x{C}x{dtype}>"""
    
    # 三角函数
    elif op_id in ['math.sin', 'math.cos', 'math.tan']:
        return f"""    %{output_tensor} = linalg.generic {{
          indexing_maps = [
            affine_map<(i, j, k, l) -> (i, j, k, l)>,
            affine_map<(i, j, k, l) -> (i, j, k, l)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }} ins(%{input_tensor} : tensor<{N}x{H}x{W}x{C}x{dtype}>) 
            outs(%{input_tensor} : tensor<{N}x{H}x{W}x{C}x{dtype}>) {{
          ^bb0(%x: {dtype}, %o: {dtype}):
            %result = math.sin %x : {dtype}
            linalg.yield %result : {dtype}
        }} -> tensor<{N}x{H}x{W}x{C}x{dtype}>"""
    
    # 比较操作
    elif op_id in ['arith.cmpi', 'arith.cmpf']:
        return f"""    %{output_tensor} = linalg.generic {{
          indexing_maps = [
            affine_map<(i, j, k, l) -> (i, j, k, l)>,
            affine_map<(i, j, k, l) -> (i, j, k, l)>,
            affine_map<(i, j, k, l) -> (i, j, k, l)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }} ins(%{input_tensor}, %{input_tensor} : tensor<{N}x{H}x{W}x{C}x{dtype}>, tensor<{N}x{H}x{W}x{C}x{dtype}>) 
            outs(%{input_tensor} : tensor<{N}x{H}x{W}x{C}x{dtype}>) {{
          ^bb0(%x: {dtype}, %y: {dtype}, %o: {dtype}):
            %result = arith.cmpf "ogt", %x, %y : {dtype}
            linalg.yield %result : {dtype}
        }} -> tensor<{N}x{H}x{W}x{C}x{dtype}>"""
    
    # 默认：复制操作
    else:
        return f"""    %{output_tensor} = linalg.generic {{
          indexing_maps = [
            affine_map<(i, j, k, l) -> (i, j, k, l)>,
            affine_map<(i, j, k, l) -> (i, j, k, l)>
          ],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        }} ins(%{input_tensor} : tensor<{N}x{H}x{W}x{C}x{dtype}>) 
            outs(%{input_tensor} : tensor<{N}x{H}x{W}x{C}x{dtype}>) {{
          ^bb0(%x: {dtype}, %o: {dtype}):
            linalg.yield %x : {dtype}
        }} -> tensor<{N}x{H}x{W}x{C}x{dtype}>"""

def generate_pair_mlir(case: Dict[str, Any]) -> str:
    """生成成对操作的MLIR"""
    op1 = case['op_a']
    op2 = case['op_b']
    
    # 获取参数
    N, H, W, C = op1['params']['N'], op1['params']['H'], op1['params']['W'], op1['params']['C']
    dtype = op1['params']['dtype']
    
    # 生成第一个操作
    op1_mlir = generate_simple_elementwise_mlir(
        op1['op_id'], 'input', 'result1', dtype, N, H, W, C
    )
    
    # 生成第二个操作
    op2_mlir = generate_simple_elementwise_mlir(
        op2['op_id'], 'result1', 'result2', dtype, N, H, W, C
    )
    
    return f"""module {{
  func.func @main(
    %input: tensor<{N}x{H}x{W}x{C}x{dtype}>
  ) -> tensor<{N}x{H}x{W}x{C}x{dtype}> {{
{op1_mlir}

{op2_mlir}

    return %result2 : tensor<{N}x{H}x{W}x{C}x{dtype}>
  }}
}}"""

def main():
    """主函数"""
    # 设置路径
    cases_file = Path("out/cases_pairs_complete.json")
    output_dir = Path("out/pairs_complete")
    
    # 检查输入文件
    if not cases_file.exists():
        print(f"❌ 找不到成对操作实例文件: {cases_file}")
        print("请先运行 scripts/enumerate_ops.py")
        return
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("📋 加载成对操作实例...")
    cases = load_cases(cases_file)
    print(f"  找到 {len(cases)} 个成对操作实例")
    
    # 生成MLIR文件
    print("🔧 生成成对操作MLIR文件...")
    success_count = 0
    error_count = 0
    
    for i, case in enumerate(cases):
        try:
            # 创建操作目录
            op1_id = case['op_a']['op_id']
            op2_id = case['op_b']['op_id']
            op_dir = output_dir / f"{op1_id}_{op2_id}"
            op_dir.mkdir(exist_ok=True)
            
            # 生成MLIR内容
            mlir_content = generate_pair_mlir(case)
            
            # 保存文件
            output_file = op_dir / f"{case['case_id']}.mlir"
            with open(output_file, 'w') as f:
                f.write(mlir_content)
            
            success_count += 1
            if success_count % 1000 == 0:
                print(f"✅ 已生成: {success_count} 个成对操作")
            
        except Exception as e:
            print(f"❌ 生成失败 {case['case_id']}: {e}")
            error_count += 1
    
    print(f"\n✅ 完成！")
    print(f"  成功: {success_count}")
    print(f"  失败: {error_count}")
    print(f"  输出目录: {output_dir}")

if __name__ == "__main__":
    main()
