"""
位置编码演示 - 改进版，清楚展示 word 的作用
"""

import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """位置编码：给每个位置添加独特的向量"""

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
print("清晰版演示：word 的实际作用")
print("="*80)

# 【关键】创建词和embedding的字典，这样word就有真正的作用
words = ["我", "爱", "你"]

# 为每个词创建一个对象，包含词和它的embedding
torch.manual_seed(42)

class WordEmbedding:
    def __init__(self, word, embedding_vector):
        self.word = word  # 词本身
        self.embedding = embedding_vector  # 该词的embedding向量

# 创建词-embedding对应表
word_data = []
for i, word in enumerate(words):
    emb = torch.randn(16)
    word_data.append(WordEmbedding(word, emb))

print("\n【第一步】原始数据：词和它的embedding")
print("-" * 80)

for item in word_data:
    print(f"\n词: '{item.word}'")
    print(f"Embedding (前5维): {item.embedding[:5].tolist()}")

# ============================================================================
# 关键：position encoding
# ============================================================================

pe_model = PositionalEncoding(d_model=16, max_seq_length=10)
pos_encoding = pe_model.pe[0]

print("\n【第二步】给每个词加上位置编码")
print("-" * 80)

results = []

for position, item in enumerate(word_data):
    # 这里 item.word 被真正使用了：
    # 1. 显示词名称
    # 2. 标记这个词在句子中的角色

    pos_code = pos_encoding[position]
    new_embedding = item.embedding + pos_code

    results.append({
        'word': item.word,           # ← word 被使用：记录词的身份
        'position': position,         # 位置
        'original_emb': item.embedding,
        'pos_code': pos_code,
        'final_emb': new_embedding
    })

    print(f"\n词: '{item.word}' (位置 {position})")
    print(f"  原始 embedding (前3维):    {item.embedding[:3].tolist()}")
    print(f"  位置编码   (前3维):       {pos_code[:3].tolist()}")
    print(f"  相加后     (前3维):       {new_embedding[:3].tolist()}")

# ============================================================================
# 验证：word 的作用
# ============================================================================

print("\n" + "="*80)
print("验证：word 的具体作用")
print("="*80)

print("\n❌ 如果没有用 word 来识别:")
print("   结果 1: [embedding_0 + pos_code_0]")
print("   结果 2: [embedding_1 + pos_code_1]")
print("   结果 3: [embedding_2 + pos_code_2]")
print("   → 不知道结果1代表哪个词，结果2代表哪个词，结果3代表哪个词")

print("\n✅ 有了 word 来识别:")
for r in results:
    print(f"   结果: {r['final_emb'][:3].tolist()} → 这是'{r['word']}'(位置{r['position']})")
print("   → 现在清楚每个结果对应哪个词！")

# ============================================================================
# 实际应用：用 word 做后续处理
# ============================================================================

print("\n" + "="*80)
print("实际应用：使用 word 做语义处理")
print("="*80)

# 建立一个词-特征映射表（模拟词汇表）
word_meanings = {
    "我": "说话者",
    "爱": "动词-正面情感",
    "你": "被说话者"
}

print("\n根据词的含义处理:")
for r in results:
    word = r['word']
    meaning = word_meanings[word]  # ← 这里 word 被用来查询词的含义
    print(f"  '{word}' (位置{r['position']}): {meaning}")
    print(f"    → 其embedding: {r['final_emb'][:3].tolist()} ...")

# ============================================================================
# 关键总结
# ============================================================================

print("\n" + "="*80)
print("关键总结：word 为什么要被使用")
print("="*80)

print("""
在这个 demo 中，word 的作用：

1️⃣  标识符作用
    word = "我"  →  识别这是第一个词
    word = "爱"  →  识别这是第二个词
    word = "你"  →  识别这是第三个词

2️⃣  查询作用
    word = "我"  →  查询字典 → 找到 "说话者"
    word = "爱"  →  查询字典 → 找到 "动词-正面情感"

3️⃣  输出作用
    print(f"'{word}'的embedding...")
    word 用来标记输出结果

4️⃣  在 Transformer 中的作用
    word_id → 词表查询 → word embedding
    word embedding + position encoding → 最终输入

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

关键认识：
position encoding 不处理词本身
position encoding 只处理位置信息

词 (word) 的处理由 embedding 层负责
位置信息的处理由 positional encoding 负责

两者加起来才是完整的输入！
""")

print("="*80)
