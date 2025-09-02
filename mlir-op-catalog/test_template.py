#!/usr/bin/env python3
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

# 设置Jinja2环境
template_dir = Path("templates")
env = Environment(
    loader=FileSystemLoader(template_dir),
    trim_blocks=True,
    lstrip_blocks=True
)

# 测试模板
template = env.get_template("linalg/elementwise_generic.mlir.j2")

# 测试数据
test_data = {
    'N': 1,
    'H': 8,
    'W': 8,
    'C': 8,
    'dtype': 'f32',
    'expr_impl': '%result = arith.maxf %a, %cst0 : f32'
}

try:
    result = template.render(**test_data)
    print("✅ 模板渲染成功:")
    print(result)
except Exception as e:
    print(f"❌ 模板渲染失败: {e}")
    import traceback
    traceback.print_exc()
