"""
Attention 机制 - 中文例子演示
用真实中文句子展示 Scaled Dot-Product Attention 的工作原理
"""

import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""

    def __init__(self, d_k: int, dropout=0.1):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: [seq_len, d_k]  或  [batch, seq_len, d_k]
        Returns:
            output: 加权的 value
            attention_weights: 注意力权重
            1. **matmul(query, key.T)**：两两 token 点积，得到两两相似度矩阵 `[seq_len, seq_len]`
        2. 除以 \(\sqrt{d_k}\)：缩放，防止维度变大时点积数值爆炸、softmax 变尖锐
        3. mask（解码器 look‑ahead 掩码）：把未来位置的分数填 `-inf`，softmax 之后变成 0，不让这个字看到后面的字
        4. softmax(dim=-1)：**每一行做 softmax**
           - 第 i 行：代表以第 i 个字作为 query，对全部 key 算概率分布，总和 = 1
           → `attention_weights [3,3]` 注意力权重矩阵
        5. dropout
        6. `output = torch.matmul(attention_weights, value)`
        """
        # 计算相似度
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 计算注意力权重
        # softmax ## softmax (scores, dim=-1) → 得到 attention_weights
        # 对**每一行单独做 softmax**，把一行的一堆分数，变成概率分布，**每行总和 = 1**。
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 加权求和
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


print("\n" + "="*90)
print("Attention 机制 - 中文句子例子")
print("="*90)

# ============================================================================
# 第1步：定义中文句子和词向量
# ============================================================================

print("\n【第1步】定义中文句子")
print("-" * 90)

sentence = "我爱自然语言处理"
words = list(sentence)

print(f"句子: {sentence}")
print(f"分词: {' '.join(words)}")
print(f"长度: {len(words)}")

# 为每个词定义词向量（真实的，不随机）
d_k = 8  # 词向量维度

# 创建词表和词向量
vocabulary = {
    "我": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    "爱": torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
    "自": torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]),
    "然": torch.tensor([2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]),
    "语": torch.tensor([1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2]),
    "言": torch.tensor([2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2]),
    "处": torch.tensor([1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8, 8.8]),
    "理": torch.tensor([2.8, 3.8, 4.8, 5.8, 6.8, 7.8, 8.8, 9.8]),
}

print(f"\n词向量维度 (d_k): {d_k}")
print(f"\n词表内容（部分）:")
for word in words[:3]:
    print(f"  '{word}': {vocabulary[word].tolist()}")

# ============================================================================
# 第2步：从词表查询所有词的向量
# ============================================================================

print(f"\n【第2步】查询句子中所有词的向量")
print("-" * 90)

# 堆栈所有词向量: [seq_len, d_k]
word_embeddings = torch.stack([vocabulary[word] for word in words])
import pdb
print(f"\n句子向量矩阵形状: {word_embeddings.shape}")
print(f"  含义: {word_embeddings.shape[0]} 个词，每个词 {word_embeddings.shape[1]} 维\n")

for i, word in enumerate(words):
    print(f"  位置{i} '{word}': {vocabulary[word].tolist()}")

# ============================================================================
# 第3步：创建 Query、Key、Value
# 在简单例子中，都用同一个词向量矩阵
# ============================================================================

print(f"\n【第3步】创建 Query、Key、Value（都用词向量）")
print("-" * 90)

query = word_embeddings      # [seq_len, d_k]
key = word_embeddings       # [seq_len, d_k]
value = word_embeddings     # [seq_len, d_k]

print(f"\nQuery 形状: {query.shape}  (每个词作为查询)")
print(f"Key 形状:   {key.shape}    (每个词作为键)")
print(f"Value 形状: {value.shape}  (每个词作为值)")

# ============================================================================
# 第4步：计算 Attention
# ============================================================================

print(f"\n【第4步】计算 Attention")
print("-" * 90)

attention = ScaledDotProductAttention(d_k=d_k)
output, attention_weights = attention(query, key, value)

print(f"\n输出形状: {output.shape}")
print(f"Attention 权重形状: {attention_weights.shape}")
"""
(Pdb) output
tensor([[ 3.1024,  4.2130,  5.3236,  6.4342,  7.5448,  8.6554,  9.7660, 10.8766],
        [ 3.1079,  4.2191,  5.3302,  6.4413,  7.5524,  8.6635,  9.7746, 10.8857],
        [ 3.1062,  4.2174,  5.3285,  6.4396,  7.5507,  8.6618,  9.7729, 10.8840],
        [ 3.1090,  4.2202,  5.3313,  6.4424,  7.5535,  8.6646,  9.7757, 10.8868],
        [ 3.1048,  4.2159,  5.3270,  6.4381,  7.5492,  8.6603,  9.7715, 10.8826],
        [ 3.1084,  4.2196,  5.3307,  6.4418,  7.5529,  8.6640,  9.7751, 10.8862],
        [ 3.1074,  4.2185,  5.3296,  6.4407,  7.5518,  8.6629,  9.7740, 10.8851],
        [ 3.1095,  4.2206,  5.3317,  6.4428,  7.5540,  8.6651,  9.7762, 10.8873]])
(Pdb) 
## 第二步：softmax (scores, dim=-1) → 得到 attention_weights
对**每一行单独做 softmax**，把一行的一堆分数，变成概率分布，**每行总和 = 1**。
attention_weights tensor([[1.2198e-10, 4.1112e-05, 7.0816e-08, 2.3867e-02, 1.5554e-09, 5.2420e-04,
         3.2243e-06, 1.0867e+00],
        [7.5995e-13, 4.3333e-06, 1.8147e-09, 1.0348e-02, 1.7061e-11, 0.0000e+00,
         1.9303e-07, 1.1007e+00],
        [9.6412e-12, 1.3365e-05, 1.1352e-08, 1.5737e-02, 1.6312e-10, 2.2613e-04,
         7.8998e-07, 1.0951e+00],
        [5.9796e-14, 1.4025e-06, 2.8959e-10, 6.7921e-03, 1.7812e-12, 4.1777e-05,
         4.7082e-08, 1.1043e+00],
        [4.4219e-11, 2.6239e-05, 3.4063e-08, 2.0213e-02, 6.3136e-10, 3.7464e-04,
         0.0000e+00, 1.0905e+00],
        [2.7493e-13, 2.7601e-06, 0.0000e+00, 8.7454e-03, 6.9112e-12, 6.9386e-05,
         1.0980e-07, 1.1023e+00],
        [2.1001e-12, 6.8015e-06, 0.0000e+00, 1.2240e-02, 4.2103e-11, 0.0000e+00,
         0.0000e+00, 1.0987e+00],
        [1.3000e-14, 7.1235e-07, 9.6233e-11, 5.2731e-03, 0.0000e+00, 2.5144e-05,
         2.0181e-08, 1.1058e+00]])

"""
# ============================================================================
# 第5步：展示 Attention 权重矩阵
# ============================================================================

print(f"\n【第5步】查看 Attention 权重矩阵（哪些词关注哪些词）")
print("-" * 90)

print(f"\nAttention 权重矩阵 (8×8):")
print(f"  行 = Query（当前词）")
print(f"  列 = Key（被关注的词）")
print(f"  值 = 注意力权重（0-1之间）\n")

# 显示权重矩阵，用词作为标签
print("     ", end="")
for word in words:
    print(f"{word:>6}", end=" ")
print()
for i, word_i in enumerate(words):
    print(f"'{word_i}' ", end="")
    for j, word_j in enumerate(words):
        weight = attention_weights[i, j].item()
        print(f"{weight:>6.3f}", end=" ")
    print()

# ============================================================================
# 第6步：分析：某个词看其他词的注意力
# ============================================================================

print(f"\n【第6步】分析：某个词关注其他词的情况】")
print("-" * 90)

# 以"我"为例（位置0）
pos_我 = 0
print(f"\n词 '{words[pos_我]}'（位置{pos_我}）的注意力分布:")
我_attention = attention_weights[pos_我]

for j, word_j in enumerate(words):
    weight = 我_attention[j].item()
    print(f"  关注 '{word_j}' (位置{j}): {weight:.4f}  ", end="")
    print("█" * int(weight * 100))  # 可视化

# 以"处"为例（位置6）
pos_处 = 6
print(f"\n词 '{words[pos_处]}'（位置{pos_处}）的注意力分布:")
处_attention = attention_weights[pos_处]

for j, word_j in enumerate(words):
    weight = 处_attention[j].item()
    print(f"  关注 '{word_j}' (位置{j}): {weight:.4f}  ", end="")
    print("█" * int(weight * 100))

# ============================================================================
# 第7步：数学讲解
# ============================================================================

print(f"\n【第7步】数学讲解：以 '我' 为例")
print("-" * 90)

print(f"""
第一步：计算相似度得分
  scores = Q · K^T / √d_k

  对于 "我"（位置0）：
    score("我"->"我") = [1,2,3,4,5,6,7,8] · [1,2,3,4,5,6,7,8] / √8
                     = (1² + 2² + 3² + ... + 8²) / √8
                     = 204 / 2.828
                     ≈ {attention_weights[0, 0].item() * 100:.1f}

    score("我"->"爱") = [1,2,3,4,5,6,7,8] · [2,3,4,5,6,7,8,9] / √8
                     = (1×2 + 2×3 + 3×4 + ... + 8×9) / √8
                     = 240 / 2.828
                     ≈ {attention_weights[0, 1].item() * 100:.1f}

    ... 依此类推

第二步：应用 Softmax 得到概率分布
  attention_weights = softmax(scores)

  "我" 的注意力分布:
    对"我": {attention_weights[0, 0].item():.4f}
    对"爱": {attention_weights[0, 1].item():.4f}
    对"自": {attention_weights[0, 2].item():.4f}
    ... 所有权重加起来 = 1.0

第三步：加权求和得到输出
  output_i = Σ (attention_weight_j × value_j)

  output("我") = {attention_weights[0, 0].item():.4f} × value("我")
               + {attention_weights[0, 1].item():.4f} × value("爱")
               + {attention_weights[0, 2].item():.4f} × value("自")
               + ...
""")

# ============================================================================
# 第8步：关键理解
# ============================================================================

print(f"\n【第8步】关键理解】")
print("="*90)

print(f"""
✅ Attention 的核心作用：

1. 【计算重要性】
   对于句子中的每个词，计算它对其他词的"关注度"

   例如："我爱自然语言处理" 中：
   • "我" 最关注 "爱"？还是最关注自己？
   • "处理" 最关注 "自然语言"？还是最关注 "处"？

2. 【加权聚合】
   根据关注度重新组合其他词的信息

   新的"我"表示 = 0.15×旧"我" + 0.20×"爱" + 0.18×"自" + ...

3. 【为什么有用？】
   允许词"看到"上下文
   • "爱" 不仅是它自己的向量
   • 还融合了周围词的信息
   • 经过 Attention 后变成更有语义的表示

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 和我们刚才的 Position Encoding 的关系：

  Transformer 输入 = Embedding + Position Encoding

  ↓（进入多头注意力）

  多头注意力 = 多个 Attention head

  • 每个 head 计算一种"关注方式"
  • head1 可能关注语法关系
  • head2 可能关注语义关系
  • 最后拼接所有 head 的输出

  ↓（然后进入 FFN）

  最终每个词都是"理解了上下文的表示"
""")
if __name__ == '__main__':
    print("="*90 + "\n")
