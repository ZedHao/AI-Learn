"""
多头注意力机制 - 中文例子演示
用真实中文句子展示 Multi-Head Attention 的工作原理
"""

import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""

    def __init__(self, d_k: int, dropout=0.0):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, value)

        return output, attention_weights


print("\n" + "="*100)
print("多头注意力 (Multi-Head Attention) - 中文句子例子")
print("="*100)

# ============================================================================
# 第1步：定义参数和中文句子
# ============================================================================

print("\n【第1步】定义多头注意力的参数")
print("-" * 100)

sentence = "我爱你"
words = list(sentence)
""" """
d_model = 8      # 总的向量维度
num_heads = 4    # 4个头
d_k = d_model // num_heads  # 每个头的维度 = 8 / 4 = 2

print(f"句子: {sentence}")
print(f"词: {' '.join(words)}")
print(f"\n多头注意力参数:")
print(f"  d_model (总维度): {d_model}")
print(f"  num_heads (头数): {num_heads}")
print(f"  d_k (每个头的维度): {d_k}")

# ============================================================================
# 第2步：定义词向量
# ============================================================================

print(f"\n【第2步】定义词向量")
print("-" * 100)

vocabulary = {
    "我": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    "爱": torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
    "你": torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
}

word_embeddings = torch.stack([vocabulary[word] for word in words])

print(f"词向量矩阵形状: {word_embeddings.shape}  [seq_len={len(words)}, d_model={d_model}]")
for i, word in enumerate(words):
    print(f"  '{word}': {vocabulary[word].tolist()}")

# ============================================================================
# 第3步：单头注意力 vs 多头注意力的对比
# ============================================================================

print(f"\n【第3步】多头注意力的思路")
print("-" * 100)

print(f"""
单头注意力（之前演示的）:
  Query [3, 8] × Key^T [8, 3] → scores [3, 3]
  → softmax → attention_weights [3, 3]
  → 加权求和 → output [3, 8]

多头注意力（现在演示的）:
  步骤1: 把 Q、K、V 从 [3, 8] 投影成 4 个头，每个头 [3, 2]
  步骤2: 每个头独立做一次 scaled dot-product attention
  步骤3: 把 4 个头的结果拼接回 [3, 8]
  步骤4: 再过一个最后的线性层（可选）
""")

# ============================================================================
# 第4步：初始化 Q、K、V 投影层
# ============================================================================

print(f"\n【第4步】初始化投影层权重")
print("-" * 100)

# 为了演示目的，创建简单的投影权重
torch.manual_seed(42)

# 投影层：[d_model, d_model] → 用来把原向量投影成不同的表示
W_q = torch.randn(d_model, d_model)
W_k = torch.randn(d_model, d_model)
W_v = torch.randn(d_model, d_model)
W_o = torch.randn(d_model, d_model)  # 最后的输出投影
import pdb
print(f"投影权重形状:")
print(f"  W_q (Query投影): {W_q.shape}")
print(f"  W_k (Key投影): {W_k.shape}")
print(f"  W_v (Value投影): {W_v.shape}")
print(f"  W_o (最终投影): {W_o.shape}")

# ============================================================================
# 第5步：投影后分头
# ============================================================================

print(f"\n【第5步】投影 Q、K、V，然后分头")
print("-" * 100)

# 投影
Q = torch.matmul(word_embeddings, W_q)  # [3, 8]
K = torch.matmul(word_embeddings, W_k)  # [3, 8]
V = torch.matmul(word_embeddings, W_v)  # [3, 8]

print(f"投影后:")
print(f"  Q: {Q.shape}")
print(f"  K: {K.shape}")
print(f"  V: {V.shape}")

# 分头：[seq_len, d_model] → [seq_len, num_heads, d_k]
"""
你原始向量是 **8 维**，`.view()` 要求元素总数严格相等。

- 原始：`[3, 8]` → 总元素数量 = 3 * 8 = **24**
- 如果写成 `view(3, 3, 3)` → 3*3*3 = **27**

> 
> 24≠27，直接抛报错：`invalid view size`。
"""
Q_heads = Q.view(len(words), num_heads, d_k)  # [3, 4, 2]
K_heads = K.view(len(words), num_heads, d_k)  # [3, 4, 2]
V_heads = V.view(len(words), num_heads, d_k)  # [3, 4, 2]

pdb.set_trace()
print(f"\n分头后:")
print(f"  Q_heads: {Q_heads.shape}  (每个头维度是 {d_k})")
print(f"  K_heads: {K_heads.shape}")
print(f"  V_heads: {V_heads.shape}")

# ============================================================================
# 第6步：每个头做一遍 attention
# ============================================================================

print(f"\n【第6步】每个头独立计算 Attention")
print("-" * 100)

attention = ScaledDotProductAttention(d_k=d_k, dropout=0.0)
head_outputs = []
head_weights = []

for head_idx in range(num_heads):
    # 提取第 head_idx 个头
    Q_h = Q_heads[:, head_idx, :]  # [3, 2]
    K_h = K_heads[:, head_idx, :]  # [3, 2]
    V_h = V_heads[:, head_idx, :]  # [3, 2]

    # 计算这个头的 attention
    output_h, weights_h = attention(Q_h, K_h, V_h)

    head_outputs.append(output_h)  # [3, 2]
    head_weights.append(weights_h)  # [3, 3]

    print(f"\n【Head {head_idx}】")
    print(f"  Q_h: {Q_h.shape}")
    print(f"  scores shape: {(Q_h @ K_h.T).shape}")
    print(f"  output_h: {output_h.shape}")
    print(f"  weights_h shape: {weights_h.shape}")

    # 展示这个头的权重矩阵
    print(f"  注意力权重矩阵:")
    print(f"         我      爱      你")
    for i, word_i in enumerate(words):
        print(f"  '{word_i}'", end="")
        for j in range(3):
            w = weights_h[i, j].item()
            print(f"  {w:6.3f}", end="")
        print()

# ============================================================================
# 第7步：拼接所有头的输出
# ============================================================================

print(f"\n【第7步】拼接所有头的输出")
print("-" * 100)

# [3, 2] + [3, 2] + [3, 2] + [3, 2] → [3, 8]
head_concat = torch.cat(head_outputs, dim=-1)  # [3, 8]

print(f"拼接后形状: {head_concat.shape}")
print(f"  原理: concat([3,2], [3,2], [3,2], [3,2]) → [3, 8]")

# ============================================================================
# 第8步：最终投影
# ============================================================================

print(f"\n【第8步】最终线性投影（可选）")
print("-" * 100)

# 通过最后的线性层
multi_head_output = torch.matmul(head_concat, W_o)  # [3, 8]

print(f"最终输出形状: {multi_head_output.shape}")
print(f"  公式: concat(所有头) × W_o")

# ============================================================================
# 第9步：总结
# ============================================================================

print(f"\n【第9步】多头注意力的关键理解】")
print("="*100)

print(f"""
📌 为什么要用多头注意力？

1. 【多个视角】
   • Head0 可能关注'我'和'爱'的关系（主语-谓语）
   • Head1 可能关注'爱'和'你'的关系（谓语-宾语）
   • Head2 可能关注'我'和'你'的关系（长距离依赖）
   • Head3 可能关注其他模式

   每个头学习不同的关注模式！

2. 【数学解释】
   单头: 一个 [3, 3] 的注意力矩阵
   多头: 4个 [3, 3] 的矩阵，每个看不同的模式
        然后拼接结果，给模型更丰富的信息

3. 【具体例子】
   对句子 "我爱你"：

   Head0 学到: "我" 最关注 "爱"（50%），然后 "你"（30%）
       →权重: ["我": 0.1, "爱": 0.5, "你": 0.3]

   Head1 学到: "我" 最关注 "你"（60%），然后 "我"自己（20%）
       →权重: ["我": 0.2, "爱": 0.1, "你": 0.6]

   Head2 学到: "我" 关注自己（40%），其他词（30%）
       →权重: ["我": 0.4, "爱": 0.3, "你": 0.3]

   Head3 学到: 更均衡的分布
       →权重: ["我": 0.33, "爱": 0.33, "你": 0.33]

   ✓ 这 4 个不同的视角被拼接在一起
   ✓ 融合成一个 8 维的最终表示

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 多头注意力 vs 单头注意力

单头:                          多头 (4头):
Q,K,V [3,8]                   Q,K,V [3,8]
   ↓                             ↓
scores [3,3]                  Head0: Q [3,2]→scores [3,3]
   ↓                          Head1: Q [3,2]→scores [3,3]
output [3,8]                  Head2: Q [3,2]→scores [3,3]
                              Head3: Q [3,2]→scores [3,3]
                                 ↓
                              concat → [3,8]
                                 ↓
                              Linear → [3,8]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 形状变化总结：

输入:
  word_embeddings: [3, 8]

投影:
  Q, K, V: [3, 8]

分头:
  Q_heads, K_heads, V_heads: [3, 4, 2]  (3个词, 4个头, 每个头2维)

每个头的 attention:
  Head0: Q[3,2] × K^T[2,3] → scores[3,3] → output[3,2]
  Head1: Q[3,2] × K^T[2,3] → scores[3,3] → output[3,2]
  Head2: Q[3,2] × K^T[2,3] → scores[3,3] → output[3,2]
  Head3: Q[3,2] × K^T[2,3] → scores[3,3] → output[3,2]

拼接:
  concat([3,2], [3,2], [3,2], [3,2]) → [3, 8]

最终投影:
  [3, 8] × [8, 8] → [3, 8]  ✓

最终输出: [3, 8]  (和输入形状一样！)
""")
if __name__ == '__main__':

    print("="*100 + "\n")
