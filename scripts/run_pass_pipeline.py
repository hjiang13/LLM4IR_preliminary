#!/usr/bin/env python3
"""
MLIR传递管道运行脚本
调用mlir-opt和mlir-translate编译MLIR到LLVM IR
"""

import argparse
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, Any

def run_command(cmd: list, timeout: int = 300) -> Dict[str, Any]:
    """运行命令并返回结果"""
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        duration = (time.time() - start_time) * 1000  # 转换为毫秒
        
        return {
            'success': result.returncode == 0,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'duration_ms': int(duration)
        }
        
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'returncode': -1,
            'stdout': '',
            'stderr': f'命令超时 (>{timeout}s)',
            'duration_ms': int((time.time() - start_time) * 1000)
        }
    except Exception as e:
        return {
            'success': False,
            'returncode': -1,
            'stdout': '',
            'stderr': str(e),
            'duration_ms': int((time.time() - start_time) * 1000)
        }

def compile_mlir_to_llvm(input_file: Path, output_file: Path, pipeline: str) -> Dict[str, Any]:
    """编译MLIR到LLVM IR"""
    
    # 选择传递管道文件
    if pipeline == 'linalg':
        pipeline_file = Path("config/pipeline_linalg_to_llvm.txt")
    elif pipeline == 'tf':
        pipeline_file = Path("config/pipeline_tf_to_llvm.txt")
    else:
        raise ValueError(f"未知的传递管道: {pipeline}")
    
    if not pipeline_file.exists():
        return {
            'success': False,
            'returncode': -1,
            'stdout': '',
            'stderr': f'传递管道文件不存在: {pipeline_file}',
            'duration_ms': 0
        }
    
    # 第一步：运行mlir-opt
    mlir_opt_cmd = [
        'mlir-opt',
        str(input_file),
        f'@{pipeline_file}',
        '-o', '/dev/stdout'
    ]
    
    print(f"🔧 运行 mlir-opt: {' '.join(mlir_opt_cmd)}")
    opt_result = run_command(mlir_opt_cmd)
    
    if not opt_result['success']:
        return opt_result
    
    # 第二步：运行mlir-translate
    mlir_translate_cmd = [
        'mlir-translate',
        '--mlir-to-llvmir',
        '-o', str(output_file)
    ]
    
    print(f"🔧 运行 mlir-translate: {' '.join(mlir_translate_cmd)}")
    
    # 通过管道传递mlir-opt的输出到mlir-translate
    try:
        start_time = time.time()
        
        # 运行mlir-opt
        opt_process = subprocess.Popen(
            mlir_opt_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 运行mlir-translate
        translate_process = subprocess.Popen(
            mlir_translate_cmd,
            stdin=opt_process.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 等待两个进程完成
        opt_process.stdout.close()
        opt_stdout, opt_stderr = opt_process.communicate()
        translate_stdout, translate_stderr = translate_process.communicate()
        
        duration = (time.time() - start_time) * 1000
        
        # 检查结果
        if opt_process.returncode != 0:
            return {
                'success': False,
                'returncode': opt_process.returncode,
                'stdout': opt_stdout,
                'stderr': opt_stderr,
                'duration_ms': int(duration)
            }
        
        if translate_process.returncode != 0:
            return {
                'success': False,
                'returncode': translate_process.returncode,
                'stdout': translate_stdout,
                'stderr': translate_stderr,
                'duration_ms': int(duration)
            }
        
        return {
            'success': True,
            'returncode': 0,
            'stdout': translate_stdout,
            'stderr': translate_stderr,
            'duration_ms': int(duration)
        }
        
    except Exception as e:
        return {
            'success': False,
            'returncode': -1,
            'stdout': '',
            'stderr': str(e),
            'duration_ms': int((time.time() - start_time) * 1000)
        }

def save_logs(input_file: Path, output_file: Path, result: Dict[str, Any], pipeline: str):
    """保存日志文件"""
    log_dir = output_file.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # 生成日志文件名
    base_name = output_file.stem
    log_file = log_dir / f"{base_name}.log.txt"
    err_file = log_dir / f"{base_name}.err.txt"
    
    # 保存成功日志
    with open(log_file, 'w') as f:
        f.write(f"输入文件: {input_file}\n")
        f.write(f"输出文件: {output_file}\n")
        f.write(f"传递管道: {pipeline}\n")
        f.write(f"执行时间: {result['duration_ms']}ms\n")
        f.write(f"返回码: {result['returncode']}\n")
        f.write(f"成功: {result['success']}\n")
        f.write(f"标准输出:\n{result['stdout']}\n")
        f.write(f"标准错误:\n{result['stderr']}\n")
    
    # 如果失败，保存错误日志
    if not result['success']:
        with open(err_file, 'w') as f:
            f.write(f"编译失败\n")
            f.write(f"返回码: {result['returncode']}\n")
            f.write(f"标准输出:\n{result['stdout']}\n")
            f.write(f"标准错误:\n{result['stderr']}\n")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MLIR传递管道运行脚本')
    parser.add_argument('--input', required=True, help='输入MLIR文件路径')
    parser.add_argument('--out-ll', required=True, help='输出LLVM IR文件路径')
    parser.add_argument('--pipeline', choices=['linalg', 'tf'], default='linalg', 
                       help='传递管道类型 (默认: linalg)')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    input_file = Path(args.input)
    output_file = Path(args.out_ll)
    
    # 检查输入文件
    if not input_file.exists():
        print(f"❌ 输入文件不存在: {input_file}")
        return 1
    
    # 创建输出目录
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if args.verbose:
        print(f"📁 输入文件: {input_file}")
        print(f"📁 输出文件: {output_file}")
        print(f"🔧 传递管道: {args.pipeline}")
    
    # 编译MLIR到LLVM IR
    result = compile_mlir_to_llvm(input_file, output_file, args.pipeline)
    
    # 保存日志
    save_logs(input_file, output_file, result, args.pipeline)
    
    # 输出结果
    if result['success']:
        print(f"✅ 编译成功: {input_file} -> {output_file}")
        print(f"   执行时间: {result['duration_ms']}ms")
        return 0
    else:
        print(f"❌ 编译失败: {input_file}")
        print(f"   返回码: {result['returncode']}")
        print(f"   错误: {result['stderr']}")
        return result['returncode']

if __name__ == "__main__":
    exit(main())
