"""
对比演示：两个不同的句子，Position Encoding 是否相同
"我爱你" vs "我很好"
"""

import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=16, max_seq_length=10):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                           -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


print("\n" + "="*100)
print("对比两个不同句子的 Position Encoding")
print("="*100)

# 创建 Position Encoding
pe_model = PositionalEncoding(d_model=16, max_seq_length=10)

# 句子1：我爱你
sentence1 = "我爱你"
words1 = list(sentence1)  # ['我', '爱', '你']

# 句子2：我很好
sentence2 = "我很好"
words2 = list(sentence2)  # ['我', '很', '好']

print(f"\n句子1: {sentence1} → {words1}")
print(f"句子2: {sentence2} → {words2}")

# ============================================================================
# 生成两个句子的 embedding（不同的）
# ============================================================================

print("\n" + "-"*100)
print("【步骤1】创建两个句子的 embedding（注意：内容不同，所以 embedding 不同）")
print("-"*100)

torch.manual_seed(42)
emb1 = torch.randn(3, 16)

torch.manual_seed(100)
emb2 = torch.randn(3, 16)

print(f"\n'我爱你' 的 embedding:")
for i, word in enumerate(words1):
    print(f"  '{word}' (位置{i}): {emb1[i][:6].tolist()} ...")

print(f"\n'我很好' 的 embedding:")
for i, word in enumerate(words2):
    print(f"  '{word}' (位置{i}): {emb2[i][:6].tolist()} ...")

print(f"\n❌ embedding 完全不同（因为词不同）")

# ============================================================================
# 提取 Position Encoding
# ============================================================================

print("\n" + "-"*100)
print("【步骤2】提取 Position Encoding（注意：这是通用的位置编码）")
print("-"*100)

pos_encoding = pe_model.pe[0]

print(f"\nPosition Encoding 矩阵:")
for i in range(3):
    print(f"  位置{i}: {pos_encoding[i][:6].tolist()} ...")

# ============================================================================
# 应用 Position Encoding
# ============================================================================

print("\n" + "-"*100)
print("【步骤3】应用 Position Encoding")
print("-"*100)

output1 = pe_model(emb1.unsqueeze(0)).squeeze(0)
output2 = pe_model(emb2.unsqueeze(0)).squeeze(0)

print(f"\n'我爱你' 加上 Position Encoding 后:")
for i, word in enumerate(words1):
    print(f"  '{word}' (位置{i}): {output1[i][:6].tolist()} ...")

print(f"\n'我很好' 加上 Position Encoding 后:")
for i, word in enumerate(words2):
    print(f"  '{word}' (位置{i}): {output2[i][:6].tolist()} ...")

# ============================================================================
# 对比 Position Encoding 部分
# ============================================================================

print("\n" + "="*100)
print("✅ 关键对比：Position Encoding 部分")
print("="*100)

print(f"\n位置编码是否完全相同？")

# 只比较 Position Encoding 部分
pos_code_1 = pe_model.pe[0][:3]
pos_code_2 = pe_model.pe[0][:3]

print(f"\n'我爱你' 的位置编码:")
print(pos_code_1)

print(f"\n'我很好' 的位置编码:")
print(pos_code_2)

print(f"\n完全相同吗？ {torch.equal(pos_code_1, pos_code_2)}")
print(f"✓ 是的，完全相同！")

# ============================================================================
# 最终对比
# ============================================================================

print("\n" + "="*100)
print("总结对比表")
print("="*100)

comparison_data = {
    "句子": ["我爱你", "我很好"],
    "词": [words1, words2],
    "Embedding 相同？": ["否", "否"],
    "Position Encoding 相同？": ["是 ✓", "是 ✓"],
    "最终输出相同？": ["否", "否"]
}

print(f"\n{'指标':<20} {'我爱你':<30} {'我很好':<30}")
print("-" * 80)

print(f"{'词':<20} {str(words1):<30} {str(words2):<30}")
print(f"{'Embedding内容':<20} {'不同':<30} {'不同':<30}")
print(f"{'Position Encoding':<20} {'[0.0, 1.0, ...]':<30} {'[0.0, 1.0, ...]':<30}")
print(f"{'Position完全相同？':<20} {'是 ✓':<30} {'（都是）':<30}")
print(f"{'最终输出':<20} {'不同':<30} {'不同':<30}")

# ============================================================================
# 数学验证
# ============================================================================

print("\n" + "="*100)
print("数学验证：为什么 Position Encoding 一定相同")
print("="*100)

print("""
Position Encoding 公式：
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

关键：公式中只有 pos（位置序号），没有词的信息！

所以：
    PE(0, ...) = sin(0) = 0.0          无论第一个词是啥，位置0的编码都是这个
    PE(1, ...) = sin(1) ≈ 0.841        无论第二个词是啥，位置1的编码都是这个
    PE(2, ...) = sin(2) ≈ 0.909        无论第三个词是啥，位置2的编码都是这个

结论：
    "我爱你" 的位置编码 = "我很好" 的位置编码 = "ABC" 的位置编码 = ...
""")

# ============================================================================
# 最后验证：差异量化
# ============================================================================

print("\n" + "="*100)
print("差异量化")
print("="*100)

# 只看 Position Encoding 部分的差异
pos_diff = torch.norm(pos_code_1 - pos_code_2)
print(f"\nPosition Encoding 差异（范数）: {pos_diff.item():.10f}")
print(f"✓ 完全相同！差异为 0")

# 看完整输出的差异
full_diff = torch.norm(output1 - output2)
print(f"\n完整输出差异（范数）: {full_diff.item():.4f}")
print(f"✗ 不同！（因为 embedding 部分不同）")

print("\n" + "="*100)
print("结论")
print("="*100)

print("""
✅ 你的理解完全正确！

Position Encoding:
  • 任何 3 个字的句子的位置编码都完全相同
  • "我爱你" 的位置编码 = "我很好" 的位置编码
  • 因为公式中只有 position 参数，没有 word 参数

这体现了 Position Encoding 的核心特性：
  📌 通用性：适用于任何内容
  📌 独立性：只编码位置信息
  📌 一致性：相同长度的句子位置编码完全相同
""")

print("="*100 + "\n")
