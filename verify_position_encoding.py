"""
验证：Position Encoding 是否对不同的3个字句子产生相同结果
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


print("\n" + "="*80)
print("验证：Position Encoding 是否只关心位置，不关心内容")
print("="*80)

# 创建 Position Encoding
pe_model = PositionalEncoding(d_model=16, max_seq_length=10)
pos_encoding = pe_model.pe[0]  # 取出编码矩阵

print("\n【实验】比较两个句子的位置编码")
print("-" * 80)

# 句子1：我 爱 你
sentence1 = "我 爱 你"
torch.manual_seed(42)
emb1 = torch.randn(3, 16)

# 句子2：我 很 好
sentence2 = "我 很 好"
torch.manual_seed(100)  # 用不同的种子，生成不同的 embedding
emb2 = torch.randn(3, 16)

print(f"\n句子1: '{sentence1}'")
print(f"句子2: '{sentence2}'")

print(f"\nembedding 是不同的:")
print(f"  句子1 embedding[0] (前3维): {emb1[0][:3].tolist()}")
print(f"  句子2 embedding[0] (前3维): {emb2[0][:3].tolist()}")
print(f"  → 两个完全不同！✗")

print(f"\n但是 Position Encoding 是相同的:")

# 提取位置编码
pos_code_0 = pos_encoding[0]  # 位置0
pos_code_1 = pos_encoding[1]  # 位置1
pos_code_2 = pos_encoding[2]  # 位置2

print(f"  位置0编码 (前3维): {pos_code_0[:3].tolist()}")
print(f"  位置1编码 (前3维): {pos_code_1[:3].tolist()}")
print(f"  位置2编码 (前3维): {pos_code_2[:3].tolist()}")

print(f"\n✅ 对两个句子都是相同的位置编码！")

# ============================================================================
# 数学证明
# ============================================================================

print("\n" + "="*80)
print("数学证明")
print("="*80)

print("""
Position Encoding 公式:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

关键点：
  • 输入只有 pos（位置序号）
  • 和 d_model（模型维度）
  • 没有 word（词的内容）

所以：
  PE(0, 0) = sin(0 / 10000^(0/16)) = sin(0) = 0.0
  PE(0, 1) = cos(0 / 10000^(0/16)) = cos(0) = 1.0
  PE(0, 2) = sin(0 / 10000^(2/16)) = sin(0) = 0.0

  PE(1, 0) = sin(1 / 10000^(0/16)) = sin(1) = 0.84...
  PE(1, 1) = cos(1 / 10000^(0/16)) = cos(1) = 0.54...
  ...

结论：无论词是什么，位置编码完全相同！
""")

# ============================================================================
# 完整对比
# ============================================================================

print("\n" + "="*80)
print("完整对比：两个句子的完整处理流程")
print("="*80)

print(f"\n句子1: '{sentence1}'")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

output1 = pe_model(emb1.unsqueeze(0)).squeeze(0)
for i, word in enumerate(sentence1.split()):
    print(f"\n'{word}' (位置{i}):")
    print(f"  原始embedding  (前3维): {emb1[i][:3].tolist()}")
    print(f"  + position编码 (前3维): {pos_encoding[i][:3].tolist()}")
    print(f"  = 最终embedding (前3维): {output1[i][:3].tolist()}")

print(f"\n\n句子2: '{sentence2}'")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

output2 = pe_model(emb2.unsqueeze(0)).squeeze(0)
for i, word in enumerate(sentence2.split()):
    print(f"\n'{word}' (位置{i}):")
    print(f"  原始embedding  (前3维): {emb2[i][:3].tolist()}")
    print(f"  + position编码 (前3维): {pos_encoding[i][:3].tolist()}")
    print(f"  = 最终embedding (前3维): {output2[i][:3].tolist()}")

# ============================================================================
# 验证关键发现
# ============================================================================

print("\n" + "="*80)
print("✅ 关键发现验证")
print("="*80)

print(f"\n【发现1】Position Encoding 部分完全相同")
print(f"  句子1位置编码: {pos_encoding[:3].tolist()}")
print(f"  句子2位置编码: {pos_encoding[:3].tolist()}")
print(f"  ✓ 完全相同")

print(f"\n【发现2】Embedding 部分不同（因为词不同）")
print(f"  句子1 embedding差异: {(emb1 - emb2).abs().sum():.2f}")

print(f"\n【发现3】最终输出不同（因为embedding不同）")
print(f"  句子1和句子2输出差异: {(output1 - output2).abs().sum():.2f}")

print(f"\n" + "="*80)
print("总结")
print("="*80)

print("""
是的，你说的完全对！✅

Position Encoding:
  • 我 爱 你 → pos encoding: [pos0, pos1, pos2]
  • 我 很 好 → pos encoding: [pos0, pos1, pos2]
  • 张 三 丰 → pos encoding: [pos0, pos1, pos2]

  → 对任何3个字的句子都完全相同！

原因：
  Position Encoding 公式中只有 position 这个参数
  没有 word 这个参数
  所以必然只关心位置，不关心内容

这就是为什么 Position Encoding 能适配任何内容的句子！
""")

print("="*80)
