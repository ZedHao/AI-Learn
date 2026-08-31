"""
Scaled Dot-Product Attention 的两个返回值详解
output 和 attention_weights 分别是什么
"""

import torch
import numpy as np

print("\n" + "="*90)
print("Attention 的两个返回值详解")
print("="*90)

# 简单例子：3个词，8维向量
words = ["我", "爱", "你"]
d_k = 8

# 词向量（简化版）
query = torch.tensor([
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],     # "我"
    [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],     # "爱"
    [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],    # "你"
])

key = query      # 简单例子中 Q=K=V
value = query

print(f"\n【输入】")
print(f"Query（查询）形状: {query.shape}")
print(f"Key（键）形状: {key.shape}")
print(f"Value（值）形状: {value.shape}")

# ============================================================================
# 手动计算 Attention（3步）
# ============================================================================

print(f"\n【第一步】计算相似度得分 scores = Q · K^T / √d_k")
print("-" * 90)

scores = torch.matmul(query, key.transpose(-1, -2)) / np.sqrt(d_k)

print(f"scores 形状: {scores.shape}  (3×3 矩阵)")
print(f"scores = \n{scores}\n")

print(f"含义:")
print(f"  scores[0,0] = '我' · '我' / √8 = {scores[0, 0].item():.4f}")
print(f"    → '我' 和 '我' 的相似度")
print(f"  scores[0,1] = '我' · '爱' / √8 = {scores[0, 1].item():.4f}")
print(f"    → '我' 和 '爱' 的相似度")
print(f"  scores[0,2] = '我' · '你' / √8 = {scores[0, 2].item():.4f}")
print(f"    → '我' 和 '你' 的相似度")

# ============================================================================
# 第二步：Softmax 得到权重
# ============================================================================

print(f"\n【第二步】应用 Softmax → attention_weights")
print("-" * 90)

attention_weights = torch.softmax(scores, dim=-1)

print(f"attention_weights 形状: {attention_weights.shape}  (3×3 矩阵)")
print(f"attention_weights = \n{attention_weights}\n")

print(f"含义（第一行 '我' 的权重）:")
print(f"  权重对 '我': {attention_weights[0, 0].item():.4f}")
print(f"  权重对 '爱': {attention_weights[0, 1].item():.4f}")
print(f"  权重对 '你': {attention_weights[0, 2].item():.4f}")
print(f"  总和: {attention_weights[0].sum().item():.4f}  ✓ = 1.0")

# ============================================================================
# 第三步：加权求和 → output
# ============================================================================

print(f"\n【第三步】加权求和：output = attention_weights · Value")
print("-" * 90)

output = torch.matmul(attention_weights, value)

print(f"output 形状: {output.shape}  (3×8 矩阵)")
print(f"output = \n{output}\n")

print(f"含义（以第一个词 "我" 为例）:")
print(f"  output[0] = attention_weights[0] · Value")
print(f"           = {attention_weights[0, 0].item():.4f} × value("我")")
print(f"             + {attention_weights[0, 1].item():.4f} × value("爱")")
print(f"             + {attention_weights[0, 2].item():.4f} × value("你")")

# ============================================================================
# 对比：输入 vs 输出
# ============================================================================

print(f"\n" + "="*90)
print("【重要对比】input 和 output 的区别")
print("="*90)

print(f"\n'我' 的向量：")
print(f"  输入（原始embedding）: {query[0].tolist()}")
print(f"  输出（经过Attention后）: {output[0].tolist()}")
print(f"  不同！✓")

print(f"\n为什么不同？")
print(f"  • 原始embedding: 只是词本身的特征")
print(f"  • 经过Attention: 融合了全句的信息")
print(f"    = {attention_weights[0, 0].item():.4f} × （'我'的信息）")
print(f"      + {attention_weights[0, 1].item():.4f} × （'爱'的信息）")
print(f"      + {attention_weights[0, 2].item():.4f} × （'你'的信息）")

# ============================================================================
# 总结：两个返回值的含义
# ============================================================================

print(f"\n" + "="*90)
print("【总结】attention() 返回的两个值")
print("="*90)

print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│ 返回值1：output                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ 形状: [seq_len, d_k]  (这里是 [3, 8])                                   │
│ 含义: 经过 Attention 加权后的"新表示"                                    │
│ 用途: 这是 Transformer 的真正输出，传给下一层（FFN）                      │
│                                                                         │
│ 特点:                                                                   │
│   • 每个词的新表示都融合了全句信息                                       │
│   • "我" 不再是 [1,2,3,...] 而是带有"爱"和"你"信息的新向量              │
│   • 可以理解为"理解了上下文的表示"                                       │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 返回值2：attention_weights                                              │
├─────────────────────────────────────────────────────────────────────────┤
│ 形状: [seq_len, seq_len]  (这里是 [3, 3])                               │
│ 含义: 注意力权重分布矩阵                                                │
│ 用途: 可视化、理解、debug                                               │
│                                                                         │
│ 特点:                                                                   │
│   • 行 = Query (查询词)，列 = Key (被关注的词)                           │
│   • 值 = 0 ~ 1 之间的权重                                               │
│   • 每一行加起来 = 1.0 (概率分布)                                       │
│                                                                         │
│ 例子:                                                                   │
│   权重[0,0] = 0.34 → "我" 关注 "我" 34%                                │
│   权重[0,1] = 0.33 → "我" 关注 "爱" 33%                                │
│   权重[0,2] = 0.33 → "我" 关注 "你" 33%                                │
└─────────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📌 实际使用场景：

1. 【训练时】
   • 只需要 output 传给下一层
   • attention_weights 可以用来计算损失或可视化

2. 【推理/分析时】
   • output: 用来生成下一个词
   • attention_weights: 用来可视化"哪个词关注哪个词"

3. 【论文/演讲中】
   • attention_weights 被广泛用来解释模型的决策
   • 比如可以看到 "处理" 最关注 "语言" 这样的词

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

数学公式总结：

  scores = Q · K^T / √d_k         [seq_len, seq_len]
    ↓
  attention_weights = softmax(scores)  [seq_len, seq_len]  ← 返回值2
    ↓
  output = attention_weights · V   [seq_len, d_k]  ← 返回值1
""")

print("="*90 + "\n")
