#!/bin/bash

# MLIR运行工具
# 用法: ./run_mlir.sh [操作名] [选项]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 显示帮助信息
show_help() {
    echo "MLIR运行工具"
    echo "============="
    echo ""
    echo "用法: $0 [操作名] [选项]"
    echo ""
    echo "操作名:"
    echo "  list                    - 列出所有可用的操作"
    echo "  run <op>               - 运行指定的操作"
    echo "  validate <op>          - 验证MLIR代码"
    echo "  optimize <op>          - 运行优化传递"
    echo "  compile <op>           - 编译为LLVM IR"
    echo "  stats                  - 显示项目统计"
    echo "  demo                   - 运行演示"
    echo ""
    echo "选项:"
    echo "  --help, -h             - 显示此帮助信息"
    echo "  --verbose, -v          - 详细输出"
    echo ""
    echo "示例:"
    echo "  $0 list"
    echo "  $0 run add"
    echo "  $0 validate matmul"
    echo "  $0 optimize relu"
    echo "  $0 compile conv2d"
}

# 列出所有可用操作
list_operations() {
    print_info "可用的MLIR操作:"
    echo ""
    
    echo "📊 单个操作 (53个):"
    find out/single -name "*.mlir" -type f | sed 's|out/single/||' | sed 's|/.*||' | sort | uniq | while read op; do
        echo "  - $op"
    done
    
    echo ""
    echo "🔗 成对操作 (96个):"
    find out/pairs_llvm -name "*.ll" -type f | head -10 | sed 's|out/pairs_llvm/||' | sed 's|/.*||' | sort | uniq | while read op; do
        echo "  - $op"
    done
    echo "  ... (还有更多)"
}

# 运行指定操作
run_operation() {
    local op_name="$1"
    local verbose="$2"
    
    # 查找操作文件
    local mlir_file=$(find out/single -name "${op_name}_*.mlir" | head -1)
    local llvm_file=$(find out/single_llvm -name "${op_name}_*.ll" | head -1)
    
    if [ -z "$mlir_file" ]; then
        print_error "未找到操作 '$op_name' 的MLIR文件"
        return 1
    fi
    
    print_info "运行操作: $op_name"
    echo "MLIR文件: $mlir_file"
    
    if [ -n "$llvm_file" ]; then
        echo "LLVM文件: $llvm_file"
    fi
    echo ""
    
    # 显示MLIR代码
    print_info "MLIR代码:"
    echo "----------------------------------------"
    cat "$mlir_file"
    echo ""
    
    # 验证MLIR
    print_info "验证MLIR代码..."
    if mlir-opt "$mlir_file" >/dev/null 2>&1; then
        print_success "MLIR代码语法正确"
    else
        print_error "MLIR代码有语法错误"
        return 1
    fi
    
    # 运行优化传递
    print_info "运行优化传递..."
    if [ "$verbose" = "true" ]; then
        mlir-opt "$mlir_file" --convert-linalg-to-loops --lower-affine --convert-scf-to-cf
    else
        mlir-opt "$mlir_file" --convert-linalg-to-loops --lower-affine --convert-scf-to-cf >/dev/null 2>&1
        print_success "优化传递完成"
    fi
    
    # 如果有LLVM文件，显示部分内容
    if [ -n "$llvm_file" ]; then
        print_info "编译后的LLVM IR (前10行):"
        echo "----------------------------------------"
        head -10 "$llvm_file"
        echo "..."
    fi
}

# 验证操作
validate_operation() {
    local op_name="$1"
    
    local mlir_file=$(find out/single -name "${op_name}_*.mlir" | head -1)
    
    if [ -z "$mlir_file" ]; then
        print_error "未找到操作 '$op_name' 的MLIR文件"
        return 1
    fi
    
    print_info "验证操作: $op_name"
    echo "文件: $mlir_file"
    echo ""
    
    if mlir-opt "$mlir_file" >/dev/null 2>&1; then
        print_success "MLIR代码语法正确"
    else
        print_error "MLIR代码有语法错误"
        return 1
    fi
}

# 优化操作
optimize_operation() {
    local op_name="$1"
    local verbose="$2"
    
    local mlir_file=$(find out/single -name "${op_name}_*.mlir" | head -1)
    
    if [ -z "$mlir_file" ]; then
        print_error "未找到操作 '$op_name' 的MLIR文件"
        return 1
    fi
    
    print_info "优化操作: $op_name"
    echo "文件: $mlir_file"
    echo ""
    
    if [ "$verbose" = "true" ]; then
        mlir-opt "$mlir_file" --convert-linalg-to-loops --lower-affine --convert-scf-to-cf
    else
        mlir-opt "$mlir_file" --convert-linalg-to-loops --lower-affine --convert-scf-to-cf >/dev/null 2>&1
        print_success "优化完成"
    fi
}

# 编译操作
compile_operation() {
    local op_name="$1"
    
    local mlir_file=$(find out/single -name "${op_name}_*.mlir" | head -1)
    local llvm_file=$(find out/single_llvm -name "${op_name}_*.ll" | head -1)
    
    if [ -z "$mlir_file" ]; then
        print_error "未找到操作 '$op_name' 的MLIR文件"
        return 1
    fi
    
    print_info "编译操作: $op_name"
    echo "MLIR文件: $mlir_file"
    
    if [ -n "$llvm_file" ]; then
        echo "LLVM文件: $llvm_file"
        print_success "已编译为LLVM IR"
        
        # 验证LLVM IR
        if llvm-as "$llvm_file" -o /tmp/test_compile.bc 2>/dev/null; then
            print_success "LLVM IR语法正确"
            rm -f /tmp/test_compile.bc
        else
            print_error "LLVM IR有语法错误"
        fi
    else
        print_warning "未找到对应的LLVM文件"
    fi
}

# 显示统计信息
show_stats() {
    print_info "项目统计信息:"
    echo ""
    
    local single_count=$(find out/single -name "*.mlir" | wc -l)
    local llvm_count=$(find out/single_llvm -name "*.ll" | wc -l)
    local pairs_count=$(find out/pairs_llvm -name "*.ll" | wc -l)
    
    echo "📊 单个操作: $single_count 个MLIR文件"
    echo "🔧 已编译: $llvm_count 个LLVM文件"
    echo "🔗 成对操作: $pairs_count 个LLVM文件"
    echo ""
    
    echo "📁 文件大小:"
    du -sh out/single out/single_llvm out/pairs_llvm 2>/dev/null || true
}

# 运行演示
run_demo() {
    print_info "运行MLIR演示..."
    echo ""
    
    # 运行add操作演示
    run_operation "add" "false"
    echo ""
    
    # 运行matmul操作演示
    run_operation "matmul" "false"
    echo ""
    
    # 显示统计
    show_stats
}

# 主函数
main() {
    local command="$1"
    local op_name="$2"
    local verbose="false"
    
    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --verbose|-v)
                verbose="true"
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                if [ -z "$command" ]; then
                    command="$1"
                elif [ -z "$op_name" ]; then
                    op_name="$1"
                fi
                shift
                ;;
        esac
    done
    
    # 检查环境
    if ! command -v mlir-opt >/dev/null 2>&1; then
        print_error "mlir-opt 未找到，请先安装MLIR工具链"
        exit 1
    fi
    
    # 执行命令
    case "$command" in
        list)
            list_operations
            ;;
        run)
            if [ -z "$op_name" ]; then
                print_error "请指定操作名"
                show_help
                exit 1
            fi
            run_operation "$op_name" "$verbose"
            ;;
        validate)
            if [ -z "$op_name" ]; then
                print_error "请指定操作名"
                show_help
                exit 1
            fi
            validate_operation "$op_name"
            ;;
        optimize)
            if [ -z "$op_name" ]; then
                print_error "请指定操作名"
                show_help
                exit 1
            fi
            optimize_operation "$op_name" "$verbose"
            ;;
        compile)
            if [ -z "$op_name" ]; then
                print_error "请指定操作名"
                show_help
                exit 1
            fi
            compile_operation "$op_name"
            ;;
        stats)
            show_stats
            ;;
        demo)
            run_demo
            ;;
        *)
            print_error "未知命令: $command"
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"
