"""
真实词表演示 - 不使用 seed，直接定义词向量
这样能真正理解 Position Encoding 和 word embedding 的独立性
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


print("\n" + "="*90)
print("真实词表演示：Position Encoding 和 Word Embedding 的完全独立性")
print("="*90)

# ============================================================================
# 创建真实词表（不用 seed，直接定义固定向量）
# ============================================================================

print("\n【第一步】创建真实词表（固定的词向量）")
print("-" * 90)

# 定义一个真实的词表（16维）
vocabulary = {
    "我": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]),
    "爱": torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]),
    "你": torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]),
    "很": torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]),
    "好": torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]),
}

print("词表内容（每个词都是固定的16维向量）：\n")
for word, vec in vocabulary.items():
    print(f"  '{word}': {vec[:8].tolist()} ... （完整16维）")

# ============================================================================
# 两个不同的句子
# ============================================================================

sentence1 = "我爱你"
sentence2 = "我很好"

print(f"\n【第二步】定义两个不同的句子")
print("-" * 90)

print(f"句子1: {sentence1}")
print(f"句子2: {sentence2}")

# ============================================================================
# 从词表中查询词向量
# ============================================================================

print(f"\n【第三步】从词表查询每个句子的词向量（不用 seed！）")
print("-" * 90)

# 句子1的embedding（从词表直接查）
emb1 = torch.stack([vocabulary[word] for word in sentence1])

# 句子2的embedding（从词表直接查）
emb2 = torch.stack([vocabulary[word] for word in sentence2])

print(f"\n句子1 '{sentence1}' 的词向量（从词表查询）:")
for i, word in enumerate(sentence1):
    print(f"  位置{i} '{word}': {emb1[i][:8].tolist()} ...")

print(f"\n句子2 '{sentence2}' 的词向量（从词表查询）:")
for i, word in enumerate(sentence2):
    print(f"  位置{i} '{word}': {emb2[i][:8].tolist()} ...")

# ============================================================================
# 创建 Position Encoding
# ============================================================================

print(f"\n【第四步】创建 Position Encoding")
print("-" * 90)

d_model = 16
pe_model = PositionalEncoding(d_model=d_model, max_seq_length=10)
pos_encoding = pe_model.pe[0]

print(f"\nPosition Encoding（通用的位置向量）:")
for i in range(3):
    print(f"  位置{i}: {pos_encoding[i][:8].tolist()} ...")

# ============================================================================
# 应用 Position Encoding
# ============================================================================

print(f"\n【第五步】词向量 + 位置编码 = Transformer 输入")
print("-" * 90)

output1 = pe_model(emb1.unsqueeze(0)).squeeze(0)
output2 = pe_model(emb2.unsqueeze(0)).squeeze(0)

print(f"\nA输入向量（'{sentence1}' + Position Encoding）:")
for i, word in enumerate(sentence1):
    print(f"  位置{i}: emb('{word}') + PE{i}")
    print(f"         {emb1[i][:8].tolist()} + {pos_encoding[i][:8].tolist()}")
    print(f"         = {output1[i][:8].tolist()} ...")

print(f"\nB输入向量（'{sentence2}' + Position Encoding）:")
for i, word in enumerate(sentence2):
    print(f"  位置{i}: emb('{word}') + PE{i}")
    print(f"         {emb2[i][:8].tolist()} + {pos_encoding[i][:8].tolist()}")
    print(f"         = {output2[i][:8].tolist()} ...")

# ============================================================================
# 对比分析
# ============================================================================

print(f"\n【第六步】对比分析：关键发现！】")
print("="*90)

print(f"\n位置0:")
print(f"  A: emb(我) + PE0 = {output1[0][:8].tolist()} ...")
print(f"  B: emb(我) + PE0 = {output2[0][:8].tolist()} ...")

pos0_same = torch.allclose(output1[0], output2[0])
print(f"  相同吗？{pos0_same}")
if pos0_same:
    print(f"  ✓ 完全相同！")
    print(f"    原因：都是用同一个词 '我' 的向量 [1, 2, 3, ...] + 同一个位置编码 PE0")

print(f"\n位置1:")
print(f"  A: emb(爱) + PE1 = {output1[1][:8].tolist()} ...")
print(f"  B: emb(很) + PE1 = {output2[1][:8].tolist()} ...")

pos1_same = torch.allclose(output1[1], output2[1])
print(f"  相同吗？{pos1_same}")
if not pos1_same:
    print(f"  ✗ 不同！")
    print(f"    原因：")
    print(f"      emb(爱) = [2, 3, 4, ...] 不等于 emb(很) = [4, 5, 6, ...]")
    print(f"      虽然 PE1 相同，但不同的词有不同的向量")

print(f"\n位置2:")
print(f"  A: emb(你) + PE2 = {output1[2][:8].tolist()} ...")
print(f"  B: emb(好) + PE2 = {output2[2][:8].tolist()} ...")

pos2_same = torch.allclose(output1[2], output2[2])
print(f"  相同吗？{pos2_same}")
if not pos2_same:
    print(f"  ✗ 不同！")
    print(f"    原因：")
    print(f"      emb(你) = [3, 4, 5, ...] 不等于 emb(好) = [5, 6, 7, ...]")
    print(f"      虽然 PE2 相同，但不同的词有不同的向量")

# ============================================================================
# 核心总结
# ============================================================================

print(f"\n" + "="*90)
print("【核心发现 - Position Encoding 的真相】")
print("="*90)

print(f"""
✅ Position Encoding 的 3 个关键特性：

1. 【只关心位置，不关心词】
   PE(position) = sin/cos(...)

   公式中只有 position 参数，没有 word 参数！
   所以：
     • PE0 对 "我" 适用 → [0.0, 1.0, 0.0, ...]
     • PE0 对 "很" 也适用 → 同样 [0.0, 1.0, 0.0, ...]
     • PE0 对任何第0位的词都适用

2. 【词向量由词表提供】
   emb(word) = vocabulary[word]

   "爱" 的向量固定是 [2.0, 3.0, 4.0, ...]
   "很" 的向量固定是 [4.0, 5.0, 6.0, ...]
   不管这个词在哪个位置，它的向量都是这样

3. 【Transformer 输入 = 词向量 + 位置编码】
   input = emb(word) + PE(position)

   • emb(word) 表达「这是什么词」
   • PE(position) 表达「这个词在第几位」
   • 两者相加才能在 Transformer 中用

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ seed 只是为了演示方便

在真实 Transformer 中：
  • 词向量：预训练好的权重矩阵，固定值
  • 位置编码：用公式算出，也是固定值
  • 都没有 seed！

我们用 seed 只是因为没有真实词表。
现在用真实词表后，就清楚看到二者的完全独立性了！
""")
if __name__ == '__main__':
    print("="*90)
