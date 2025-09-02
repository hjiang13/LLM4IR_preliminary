#!/usr/bin/env python3
"""
成对操作编译脚本
编译所有20,164个成对操作到LLVM IR
"""

import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any

def compile_mlir_to_llvm(mlir_file: Path, output_dir: Path) -> Dict[str, Any]:
    """编译单个MLIR文件到LLVM IR"""
    try:
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成输出文件名
        output_file = output_dir / f"{mlir_file.stem}.ll"
        
        # 第一步：使用mlir-opt应用pass pipeline
        pipeline_text = """builtin.module(
          one-shot-bufferize{bufferize-function-boundaries},
          convert-linalg-to-loops,
          lower-affine,
          convert-scf-to-cf,
          convert-cf-to-llvm,
          convert-arith-to-llvm,
          convert-math-to-llvm,
          convert-func-to-llvm,
          finalize-memref-to-llvm,
          reconcile-unrealized-casts
        )"""
        
        cmd1 = [
            "mlir-opt",
            str(mlir_file),
            f"--pass-pipeline={pipeline_text}",
            "-o", str(output_file)
        ]
        
        result1 = subprocess.run(
            cmd1,
            capture_output=True,
            text=True,
            timeout=30  # 30秒超时
        )
        
        if result1.returncode != 0:
            return {
                'status': 'failed',
                'error': f"mlir-opt failed: {result1.stderr}",
                'return_code': result1.returncode,
                'execution_time': 0
            }
        
        # 第二步：使用mlir-translate转换为LLVM IR
        cmd2 = [
            "mlir-translate",
            str(output_file),
            "--mlir-to-llvmir",
            "-o", str(output_file)
        ]
        
        result2 = subprocess.run(
            cmd2,
            capture_output=True,
            text=True,
            timeout=30  # 30秒超时
        )
        
        if result2.returncode != 0:
            return {
                'status': 'failed',
                'error': f"mlir-translate failed: {result2.stderr}",
                'return_code': result2.returncode,
                'execution_time': 0
            }
        
        return {
            'status': 'success',
            'error': None,
            'return_code': 0,
            'execution_time': 0
        }
        
    except subprocess.TimeoutExpired:
        return {
            'status': 'timeout',
            'error': 'Compilation timeout (30s)',
            'return_code': -1,
            'execution_time': 30
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'return_code': -1,
            'execution_time': 0
        }

def compile_all_pairs():
    """编译所有成对操作"""
    # 设置路径
    input_dir = Path("out/pairs_complete")
    output_dir = Path("out/pairs_complete_llvm")
    
    # 检查输入目录
    if not input_dir.exists():
        print("❌ 成对操作MLIR目录不存在")
        return
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有MLIR文件
    mlir_files = list(input_dir.rglob("*.mlir"))
    print(f"📋 找到 {len(mlir_files)} 个MLIR文件")
    print(f"🔧 开始编译到LLVM IR...")
    print()
    
    # 编译统计
    success_count = 0
    failed_count = 0
    timeout_count = 0
    error_count = 0
    
    start_time = time.time()
    
    for i, mlir_file in enumerate(mlir_files):
        try:
            # 计算相对路径以保持目录结构
            rel_path = mlir_file.relative_to(input_dir)
            output_subdir = output_dir / rel_path.parent
            
            # 编译
            result = compile_mlir_to_llvm(mlir_file, output_subdir)
            
            # 统计结果
            if result['status'] == 'success':
                success_count += 1
            elif result['status'] == 'failed':
                failed_count += 1
            elif result['status'] == 'timeout':
                timeout_count += 1
            else:
                error_count += 1
            
            # 进度显示
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(mlir_files) - i - 1) / rate if rate > 0 else 0
                
                print(f"📊 进度: {i+1:5d}/{len(mlir_files):5d} "
                      f"({(i+1)/len(mlir_files)*100:5.1f}%) "
                      f"成功: {success_count:4d} "
                      f"失败: {failed_count:4d} "
                      f"超时: {timeout_count:4d} "
                      f"错误: {error_count:4d} "
                      f"速率: {rate:.1f} 文件/秒 "
                      f"预计剩余: {eta/60:.1f} 分钟")
            
        except Exception as e:
            print(f"❌ 处理文件 {mlir_file} 时出错: {e}")
            error_count += 1
    
    # 最终统计
    total_time = time.time() - start_time
    success_rate = success_count / len(mlir_files) * 100 if mlir_files else 0
    
    print(f"\n✅ 编译完成！")
    print(f"⏱️  总耗时: {total_time/60:.1f} 分钟")
    print(f"📊 编译结果:")
    print(f"  总文件数: {len(mlir_files):,}")
    print(f"  成功: {success_count:,} ({success_rate:.1f}%)")
    print(f"  失败: {failed_count:,}")
    print(f"  超时: {timeout_count:,}")
    print(f"  错误: {error_count:,}")
    print(f"  输出目录: {output_dir}")
    
    # 保存编译报告
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_files': len(mlir_files),
        'success_count': success_count,
        'failed_count': failed_count,
        'timeout_count': timeout_count,
        'error_count': error_count,
        'success_rate': success_rate,
        'total_time_minutes': total_time / 60,
        'input_directory': str(input_dir),
        'output_directory': str(output_dir)
    }
    
    report_file = Path("out/reports/pairs_compilation_report.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, 'w') as f:
        import json
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📊 编译报告已保存到: {report_file}")

if __name__ == "__main__":
    compile_all_pairs()
