#!/usr/bin/env python3
"""
简化版MLIR编译脚本
编译所有142个操作到LLVM IR
"""

import subprocess
import time
from pathlib import Path

def compile_mlir_to_llvm(mlir_file: Path, output_file: Path, pipeline: str) -> dict:
    """编译MLIR到LLVM IR"""
    start_time = time.time()
    
    try:
        # 确保输出目录存在
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 第一步：运行mlir-opt
        cmd1 = [
            'mlir-opt',
            str(mlir_file),
            '--pass-pipeline=' + pipeline,
            '-o', str(output_file.with_suffix('.opt.mlir'))
        ]
        
        result1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=30)
        
        if result1.returncode != 0:
            return {
                'success': False,
                'error': f"mlir-opt失败: {result1.stderr}",
                'execution_time': time.time() - start_time
            }
        
        # 第二步：运行mlir-translate
        cmd2 = [
            'mlir-translate',
            str(output_file.with_suffix('.opt.mlir')),
            '--mlir-to-llvmir',
            '-o', str(output_file)
        ]
        
        result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=30)
        
        if result2.returncode != 0:
            return {
                'success': False,
                'error': f"mlir-translate失败: {result2.stderr}",
                'execution_time': time.time() - start_time
            }
        
        # 清理中间文件
        output_file.with_suffix('.opt.mlir').unlink(missing_ok=True)
        
        return {
            'success': True,
            'error': '',
            'execution_time': time.time() - start_time
        }
        
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': '执行超时',
            'execution_time': time.time() - start_time
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'execution_time': time.time() - start_time
        }

def main():
    """主函数"""
    # 设置路径
    input_dir = Path("out/single_complete")
    output_dir = Path("out/single_complete_llvm")
    pipeline_file = Path("config/pipeline_linalg_to_llvm.txt")
    
    # 检查输入目录
    if not input_dir.exists():
        print(f"❌ 找不到输入目录: {input_dir}")
        print("请先运行 scripts/gen_singletons.py")
        return
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
            
            # 编译
            result = compile_mlir_to_llvm(mlir_file, output_file, pipeline)
            
            if result['success']:
                success_count += 1
                if success_count % 10 == 0:
                    print(f"✅ 已编译: {success_count} 个操作")
            else:
                error_count += 1
                print(f"❌ 编译失败: {mlir_file.name} - {result['error']}")
            
        except Exception as e:
            error_count += 1
            print(f"❌ 处理失败: {mlir_file.name} - {e}")
    
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

if __name__ == "__main__":
    main()
