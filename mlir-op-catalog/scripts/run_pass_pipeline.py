#!/usr/bin/env python3
"""
MLIR Pass Pipeline执行脚本
执行MLIR到LLVM IR的转换
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging

def setup_logging(log_dir: Path):
    """设置日志"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'compilation.log'),
            logging.StreamHandler()
        ]
    )

def load_pipeline_config(pipeline_file: Path) -> str:
    """加载Pass Pipeline配置"""
    with open(pipeline_file, 'r') as f:
        return f.read().strip()

def run_mlir_opt(mlir_file: Path, pipeline: str, output_file: Path) -> Tuple[bool, str, float]:
    """运行mlir-opt"""
    start_time = time.time()
    
    try:
        # 构建命令
        cmd = [
            'mlir-opt',
            str(mlir_file),
            '--pass-pipeline=' + pipeline,
            '-o', str(output_file)
        ]
        
        # 执行命令
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30  # 30秒超时
        )
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            return True, "", execution_time
        else:
            return False, result.stderr, execution_time
            
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        return False, "执行超时", execution_time
    except Exception as e:
        execution_time = time.time() - start_time
        return False, str(e), execution_time

def run_mlir_translate(mlir_file: Path, output_file: Path) -> Tuple[bool, str, float]:
    """运行mlir-translate"""
    start_time = time.time()
    
    try:
        # 构建命令
        cmd = [
            'mlir-translate',
            str(mlir_file),
            '--mlir-to-llvmir',
            '-o', str(output_file)
        ]
        
        # 执行命令
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30  # 30秒超时
        )
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            return True, "", execution_time
        else:
            return False, result.stderr, execution_time
            
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        return False, "执行超时", execution_time
    except Exception as e:
        execution_time = time.time() - start_time
        return False, str(e), execution_time

def run_llvm_as(ll_file: Path, output_file: Path) -> Tuple[bool, str, float]:
    """运行llvm-as"""
    start_time = time.time()
    
    try:
        # 构建命令
        cmd = [
            'llvm-as',
            str(ll_file),
            '-o', str(output_file)
        ]
        
        # 执行命令
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30  # 30秒超时
        )
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            return True, "", execution_time
        else:
            return False, result.stderr, execution_time
            
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        return False, "执行超时", execution_time
    except Exception as e:
        execution_time = time.time() - start_time
        return False, str(e), execution_time

def compile_mlir_to_llvm(mlir_file: Path, pipeline: str, output_dir: Path) -> Dict[str, Any]:
    """编译MLIR到LLVM IR"""
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成输出文件名
    base_name = mlir_file.stem
    mlir_opt_output = output_dir / f"{base_name}_opt.mlir"
    llvm_ir_output = output_dir / f"{base_name}.ll"
    bitcode_output = output_dir / f"{base_name}.bc"
    
    results = {}
    
    # 步骤1: mlir-opt
    logging.info(f"🔧 步骤1: mlir-opt - {mlir_file.name}")
    success, error, time_taken = run_mlir_opt(mlir_file, pipeline, mlir_opt_output)
    
    results['mlir_opt'] = {
        'success': success,
        'error': error,
        'time': time_taken,
        'output_file': str(mlir_opt_output)
    }
    
    if not success:
        logging.error(f"❌ mlir-opt失败: {error}")
        return results
    
    # 步骤2: mlir-translate
    logging.info(f"🔧 步骤2: mlir-translate - {mlir_file.name}")
    success, error, time_taken = run_mlir_translate(mlir_opt_output, llvm_ir_output)
    
    results['mlir_translate'] = {
        'success': success,
        'error': error,
        'time': time_taken,
        'output_file': str(llvm_ir_output)
    }
    
    if not success:
        logging.error(f"❌ mlir-translate失败: {error}")
        return results
    
    # 步骤3: llvm-as
    logging.info(f"🔧 步骤3: llvm-as - {mlir_file.name}")
    success, error, time_taken = run_llvm_as(llvm_ir_output, bitcode_output)
    
    results['llvm_as'] = {
        'success': success,
        'error': error,
        'time': time_taken,
        'output_file': str(bitcode_output)
    }
    
    if not success:
        logging.error(f"❌ llvm-as失败: {error}")
        return results
    
    logging.info(f"✅ 编译成功: {mlir_file.name}")
    return results

def process_directory(input_dir: Path, pipeline: str, output_dir: Path, log_dir: Path) -> Dict[str, Any]:
    """处理目录中的所有MLIR文件"""
    # 设置日志
    setup_logging(log_dir)
    
    # 查找所有MLIR文件
    mlir_files = list(input_dir.rglob("*.mlir"))
    
    if not mlir_files:
        logging.warning(f"在 {input_dir} 中未找到MLIR文件")
        return {}
    
    logging.info(f"🔍 找到 {len(mlir_files)} 个MLIR文件")
    
    # 处理每个文件
    results = {}
    success_count = 0
    error_count = 0
    
    for mlir_file in mlir_files:
        try:
            # 创建相对输出目录
            rel_path = mlir_file.relative_to(input_dir)
            file_output_dir = output_dir / rel_path.parent / mlir_file.stem
            
            # 编译文件
            result = compile_mlir_to_llvm(mlir_file, pipeline, file_output_dir)
            
            # 记录结果
            results[str(mlir_file)] = result
            
            # 统计成功/失败
            if result.get('llvm_as', {}).get('success', False):
                success_count += 1
            else:
                error_count += 1
                
        except Exception as e:
            logging.error(f"❌ 处理文件失败 {mlir_file}: {e}")
            error_count += 1
            results[str(mlir_file)] = {'error': str(e)}
    
    # 保存结果
    summary = {
        'total_files': len(mlir_files),
        'success_count': success_count,
        'error_count': error_count,
        'success_rate': success_count / len(mlir_files) if mlir_files else 0,
        'results': results
    }
    
    with open(log_dir / 'compilation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"📊 编译完成:")
    logging.info(f"  总文件数: {len(mlir_files)}")
    logging.info(f"  成功: {success_count}")
    logging.info(f"  失败: {error_count}")
    logging.info(f"  成功率: {summary['success_rate']:.2%}")
    
    return summary

def main():
    """主函数"""
    # 设置路径
    input_dir = Path("out/single_complete")
    output_dir = Path("out/single_complete_llvm")
    pipeline_file = Path("config/pipeline_linalg_to_llvm.txt")
    log_file = Path("out/logs/single_complete_compilation.log")
    
    # 检查输入目录
    if not input_dir.exists():
        print(f"❌ 找不到输入目录: {input_dir}")
        print("请先运行 scripts/gen_singletons.py")
        return
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查编译管道文件
    if not pipeline_file.exists():
        print(f"❌ 找不到编译管道文件: {pipeline_file}")
        return
    
    # 加载编译管道
    with open(pipeline_file, 'r') as f:
        pipeline = f.read().strip()
    
    print(f"📋 使用编译管道: {pipeline}")
    
    # 获取所有MLIR文件
    mlir_files = list(input_dir.rglob("*.mlir"))
    print(f"🔍 找到 {len(mlir_files)} 个MLIR文件")
    
    # 编译统计
    success_count = 0
    error_count = 0
    
    # 开始编译
    print("🔧 开始编译...")
    start_time = time.time()
    
    for i, mlir_file in enumerate(mlir_files):
        try:
            # 创建输出目录结构
            relative_path = mlir_file.relative_to(input_dir)
            output_file = output_dir / relative_path.with_suffix('.ll')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 编译
            result = compile_mlir_to_llvm(mlir_file, output_file, pipeline)
            
            if result['success']:
                success_count += 1
                if success_count % 10 == 0:
                    print(f"✅ 已编译: {success_count} 个操作")
            else:
                error_count += 1
                print(f"❌ 编译失败: {mlir_file.name} - {result['error']}")
            
            # 记录日志
            log_compilation_result(log_file, mlir_file, result)
            
        except Exception as e:
            error_count += 1
            print(f"❌ 处理失败: {mlir_file.name} - {e}")
            log_compilation_result(log_file, mlir_file, {
                'success': False,
                'error': str(e),
                'execution_time': 0
            })
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 输出结果
    print(f"\n✅ 编译完成！")
    print(f"  总文件数: {len(mlir_files)}")
    print(f"  成功: {success_count}")
    print(f"  失败: {error_count}")
    print(f"  成功率: {success_count/len(mlir_files)*100:.1f}%")
    print(f"  总耗时: {total_time:.1f} 秒")
    print(f"  输出目录: {output_dir}")
    print(f"  日志文件: {log_file}")

if __name__ == "__main__":
    main()
