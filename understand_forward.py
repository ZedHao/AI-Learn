"""
演示：forward 函数是如何被调用的
"""

import torch
import torch.nn as nn


class MyModule(nn.Module):
    """自定义模块，用来演示 forward 的调用"""

    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x):
        print(f"✓ forward 方法被调用了！")
        print(f"  模块名: {self.name}")
        print(f"  输入形状: {x.shape}")
        return x * 2  # 简单的处理：乘以2


print("\n" + "="*80)
print("演示：forward 方法的自动调用")
print("="*80)

# 创建模块
model = MyModule(name="我的模块")

print("\n【方式1】用 () 调用（最常用）")
print("-" * 80)

x = torch.randn(2, 3)
print(f"输入: {x}")

print("\n执行: output = model(x)")
output1 = model(x)

print(f"输出: {output1}")

print("\n【方式2】手动调用 forward（等价的）")
print("-" * 80)

print("\n执行: output = model.forward(x)")
output2 = model.forward(x)

print(f"输出: {output2}")

print("\n【结果对比】")
print("-" * 80)

print(f"\n方式1结果 == 方式2结果: {torch.equal(output1, output2)}")
print(f"✓ 两种方式完全相同！")

print("\n" + "="*80)
print("关键认识")
print("="*80)

print("""
当你写:
    output = model(x)

PyTorch 会：
    1. 检查 model 是否继承了 nn.Module
    2. 如果是，自动调用 model.forward(x)
    3. 返回 forward 的结果

所以 forward 方法虽然"看不见"，但一直在被调用！

这叫做 "魔法方法"（Magic Method）
在 Python 中用 __call__ 实现
""")

print("="*80)
