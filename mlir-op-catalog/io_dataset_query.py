#!/usr/bin/env python3
"""
输入-输出-IR数据集查询工具
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

class IODatasetQuery:
    """输入输出数据集查询工具"""
    
    def __init__(self, dataset_file: str = "io_dataset/complete_io_dataset.json"):
        self.dataset_file = dataset_file
        self.dataset = self.load_dataset()
    
    def load_dataset(self) -> Dict:
        """加载数据集"""
        try:
            with open(self.dataset_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ 数据集文件未找到: {self.dataset_file}")
            return {}
        except Exception as e:
            print(f"❌ 加载数据集失败: {e}")
            return {}
    
    def get_operation(self, operation_name: str) -> Optional[Dict]:
        """获取特定操作的数据"""
        return self.dataset.get("operations", {}).get(operation_name)
    
    def list_operations(self, operation_type: str = None, is_pair: bool = None) -> List[str]:
        """列出操作"""
        operations = []
        for op_name, op_data in self.dataset.get("operations", {}).items():
            if operation_type and op_data.get("operation_type") != operation_type:
                continue
            if is_pair is not None and op_data.get("is_pair") != is_pair:
                continue
            operations.append(op_name)
        return operations
    
    def get_operation_types(self) -> List[str]:
        """获取所有操作类型"""
        types = set()
        for op_data in self.dataset.get("operations", {}).values():
            types.add(op_data.get("operation_type", "unknown"))
        return sorted(list(types))
    
    def get_shape_distribution(self) -> Dict[str, int]:
        """获取形状分布"""
        shapes = {}
        for op_data in self.dataset.get("operations", {}).values():
            for shape in op_data.get("input_shapes", []):
                shape_key = "x".join(map(str, shape))
                shapes[shape_key] = shapes.get(shape_key, 0) + 1
        return shapes
    
    def search_operations(self, keyword: str) -> List[str]:
        """搜索操作"""
        results = []
        keyword_lower = keyword.lower()
        for op_name, op_data in self.dataset.get("operations", {}).items():
            if (keyword_lower in op_name.lower() or 
                keyword_lower in op_data.get("operation_type", "").lower()):
                results.append(op_name)
        return results
    
    def get_data_statistics(self) -> Dict:
        """获取数据统计信息"""
        return self.dataset.get("statistics", {})
    
    def get_metadata(self) -> Dict:
        """获取元数据"""
        return self.dataset.get("metadata", {})
    
    def validate_operation_data(self, operation_name: str) -> Dict:
        """验证操作数据的正确性"""
        op_data = self.get_operation(operation_name)
        if not op_data:
            return {"valid": False, "error": "Operation not found"}
        
        validation_result = {
            "operation_name": operation_name,
            "valid": True,
            "checks": {},
            "errors": []
        }
        
        # 检查输入数据
        input_data = op_data.get("input_data", [])
        input_shapes = op_data.get("input_shapes", [])
        
        if len(input_data) != len(input_shapes):
            validation_result["valid"] = False
            validation_result["errors"].append("Input data length mismatch")
        
        for i, (data, shape) in enumerate(zip(input_data, input_shapes)):
            # 将嵌套列表转换为numpy数组来计算实际大小
            data_array = np.array(data)
            if data_array.size != np.prod(shape):
                validation_result["valid"] = False
                validation_result["errors"].append(f"Input {i} size mismatch: expected {np.prod(shape)}, got {data_array.size}")
        
        # 检查输出数据
        output_data = op_data.get("output_data", [])
        output_shape = op_data.get("output_shape", [])
        
        output_array = np.array(output_data)
        if output_array.size != np.prod(output_shape):
            validation_result["valid"] = False
            validation_result["errors"].append(f"Output data size mismatch: expected {np.prod(output_shape)}, got {output_array.size}")
        
        # 检查数据范围
        data_stats = op_data.get("data_statistics", {})
        if data_stats:
            validation_result["checks"]["data_ranges_valid"] = True
            for i, input_range in enumerate(data_stats.get("input_ranges", [])):
                if input_range["min"] > input_range["max"]:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Input {i} range invalid")
        
        return validation_result
    
    def export_operation_data(self, operation_name: str, format: str = "numpy") -> Optional[Any]:
        """导出操作数据"""
        op_data = self.get_operation(operation_name)
        if not op_data:
            return None
        
        if format == "numpy":
            input_arrays = [np.array(data) for data in op_data.get("input_data", [])]
            output_array = np.array(op_data.get("output_data", []))
            return {
                "input_arrays": input_arrays,
                "output_array": output_array,
                "input_shapes": op_data.get("input_shapes", []),
                "output_shape": op_data.get("output_shape", [])
            }
        elif format == "json":
            return op_data
        else:
            return None
    
    def compare_operations(self, op1_name: str, op2_name: str) -> Dict:
        """比较两个操作"""
        op1 = self.get_operation(op1_name)
        op2 = self.get_operation(op2_name)
        
        if not op1 or not op2:
            return {"error": "One or both operations not found"}
        
        comparison = {
            "operation1": op1_name,
            "operation2": op2_name,
            "similarities": [],
            "differences": []
        }
        
        # 比较操作类型
        if op1.get("operation_type") == op2.get("operation_type"):
            comparison["similarities"].append("Same operation type")
        else:
            comparison["differences"].append(f"Different operation types: {op1.get('operation_type')} vs {op2.get('operation_type')}")
        
        # 比较形状
        if op1.get("input_shapes") == op2.get("input_shapes"):
            comparison["similarities"].append("Same input shapes")
        else:
            comparison["differences"].append("Different input shapes")
        
        if op1.get("output_shape") == op2.get("output_shape"):
            comparison["similarities"].append("Same output shape")
        else:
            comparison["differences"].append("Different output shape")
        
        # 比较数据统计
        stats1 = op1.get("data_statistics", {})
        stats2 = op2.get("data_statistics", {})
        
        if stats1 and stats2:
            output_range1 = stats1.get("output_range", {})
            output_range2 = stats2.get("output_range", {})
            
            if (abs(output_range1.get("mean", 0) - output_range2.get("mean", 0)) < 0.1):
                comparison["similarities"].append("Similar output means")
            else:
                comparison["differences"].append("Different output means")
        
        return comparison
    
    def generate_summary_report(self) -> str:
        """生成汇总报告"""
        metadata = self.get_metadata()
        stats = self.get_data_statistics()
        
        report = f"""
# 输入-输出-IR数据集汇总报告

## 基本信息
- **创建时间**: {metadata.get('created_at', 'N/A')}
- **版本**: {metadata.get('version', 'N/A')}
- **总操作数**: {metadata.get('total_operations', 0):,}
- **单个操作数**: {metadata.get('single_operations', 0):,}
- **成对操作数**: {metadata.get('pair_operations', 0):,}

## 统计信息
- **总元素数**: {stats.get('total_operations', 0):,}
- **平均每操作元素数**: {stats.get('complexity_metrics', {}).get('average_elements_per_operation', 0):.1f}
- **估计内存使用**: {stats.get('complexity_metrics', {}).get('memory_usage_estimate_mb', 0):.2f} MB

## 操作类型分布
"""
        
        operation_types = self.get_operation_types()
        for op_type in operation_types[:10]:  # 只显示前10个
            count = stats.get('operation_types', {}).get(op_type, 0)
            report += f"- **{op_type}**: {count}\n"
        
        if len(operation_types) > 10:
            report += f"- ... 还有 {len(operation_types) - 10} 个操作类型\n"
        
        return report

def main():
    """主函数 - 命令行接口"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python3 io_dataset_query.py <command> [args...]")
        print("命令:")
        print("  list [--type TYPE] [--pair] [--single]")
        print("  get <operation_name>")
        print("  search <keyword>")
        print("  validate <operation_name>")
        print("  compare <op1> <op2>")
        print("  stats")
        print("  summary")
        return
    
    query = IODatasetQuery()
    command = sys.argv[1]
    
    if command == "list":
        operation_type = None
        is_pair = None
        
        for i, arg in enumerate(sys.argv[2:], 2):
            if arg == "--type" and i + 1 < len(sys.argv):
                operation_type = sys.argv[i + 1]
            elif arg == "--pair":
                is_pair = True
            elif arg == "--single":
                is_pair = False
        
        operations = query.list_operations(operation_type, is_pair)
        print(f"找到 {len(operations)} 个操作:")
        for op in operations:
            print(f"  - {op}")
    
    elif command == "get":
        if len(sys.argv) < 3:
            print("用法: get <operation_name>")
            return
        
        op_name = sys.argv[2]
        op_data = query.get_operation(op_name)
        if op_data:
            print(f"操作: {op_name}")
            print(f"类型: {op_data.get('operation_type')}")
            print(f"是否成对: {op_data.get('is_pair')}")
            print(f"输入形状: {op_data.get('input_shapes')}")
            print(f"输出形状: {op_data.get('output_shape')}")
        else:
            print(f"操作 '{op_name}' 未找到")
    
    elif command == "search":
        if len(sys.argv) < 3:
            print("用法: search <keyword>")
            return
        
        keyword = sys.argv[2]
        results = query.search_operations(keyword)
        print(f"搜索 '{keyword}' 找到 {len(results)} 个结果:")
        for op in results:
            print(f"  - {op}")
    
    elif command == "validate":
        if len(sys.argv) < 3:
            print("用法: validate <operation_name>")
            return
        
        op_name = sys.argv[2]
        result = query.validate_operation_data(op_name)
        print(f"验证结果: {'✅ 有效' if result['valid'] else '❌ 无效'}")
        if result.get('errors'):
            print("错误:")
            for error in result['errors']:
                print(f"  - {error}")
    
    elif command == "compare":
        if len(sys.argv) < 4:
            print("用法: compare <op1> <op2>")
            return
        
        op1, op2 = sys.argv[2], sys.argv[3]
        result = query.compare_operations(op1, op2)
        if 'error' in result:
            print(f"错误: {result['error']}")
        else:
            print(f"比较 {op1} 和 {op2}:")
            print("相似点:")
            for sim in result.get('similarities', []):
                print(f"  ✅ {sim}")
            print("差异点:")
            for diff in result.get('differences', []):
                print(f"  ❌ {diff}")
    
    elif command == "stats":
        stats = query.get_data_statistics()
        print("数据统计:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    elif command == "summary":
        print(query.generate_summary_report())
    
    else:
        print(f"未知命令: {command}")

if __name__ == "__main__":
    main()
