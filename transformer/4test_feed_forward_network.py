"""
前馈网络 (Feed Forward Network, FFN) - 中文例子演示
用真实中文句子展示 Feed Forward Network 的工作原理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


print("\n" + "=" * 100)
print("前馈网络 (Feed Forward Network) - 中文句子例子")
print("=" * 100)

# ============================================================================
# 第1步：理解 FFN 的概念
# ============================================================================

print("\n【第1步】理解 FFN 的概念")
print("-" * 100)

print(f"""
Transformer 中的前馈网络结构:
    FFN(x) = Linear_2(ReLU(Linear_1(x)))

具体过程:
  1. 输入 x [seq_len, d_model]
  2. 第一层线性投影：扩展维度 → [seq_len, d_ff]
     通常 d_ff = 4 * d_model (扩展4倍)
  3. 加入 ReLU 激活函数：引入非线性
  4. 第二层线性投影：降回原维度 → [seq_len, d_model]
  5. 输出 [seq_len, d_model]

作用:
  - 第一层（扩展）: 增加模型容量，学习复杂的特征组合
  - ReLU: 引入非线性，让模型能学习非线性函数
  - 第二层（投影回）: 降维回到原始空间，准备下一层
  - Dropout: 防止过拟合
""")

# ============================================================================
# 第2步：定义参数和中文句子
# ============================================================================

print("\n【第2步】定义参数和中文句子")
print("-" * 100)

sentence = "我爱自然语言处理"
words = list(sentence)
seq_len = len(words)
d_model = 8      # 模型维度
d_ff = 32        # 前馈网络的中间维度（扩展维度），通常是 d_model 的 4 倍

print(f"句子: {sentence}")
print(f"词: {' '.join(words)}")
print(f"\n参数设置:")
print(f"  seq_len (序列长度): {seq_len}")
print(f"  d_model (模型维度): {d_model}")
print(f"  d_ff (前馈中间维度): {d_ff}")
print(f"  扩展倍数: {d_ff / d_model}x")

# ============================================================================
# 第3步：创建词向量
# ============================================================================

print(f"\n【第3步】创建词向量")
print("-" * 100)

# 简化的词汇表
vocabulary = {
    "我": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    "爱": torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
    "自": torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
    "然": torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]),
    "语": torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
    "言": torch.tensor([6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]),
    "处": torch.tensor([7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]),
    "理": torch.tensor([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]),
}

# 获取句子的词向量
input_embeddings = torch.stack([vocabulary[word] for word in words])
print(f"输入词向量矩阵形状: {input_embeddings.shape}  [seq_len={seq_len}, d_model={d_model}]")
print(f"\n前3个词的向量:")
for i in range(min(3, len(words))):
    print(f"  '{words[i]}': {input_embeddings[i].tolist()}")

# ============================================================================
# 第4步：手动实现 FFN 的详细过程
# ============================================================================

print(f"\n【第4步】手动实现 FFN 的详细过程")
print("-" * 100)

# 初始化两个线性层
linear1_weight = torch.randn(d_ff, d_model) * 0.1  # [d_ff, d_model]
linear1_bias = torch.randn(d_ff) * 0.01              # [d_ff]

linear2_weight = torch.randn(d_model, d_ff) * 0.1  # [d_model, d_ff]
linear2_bias = torch.randn(d_model) * 0.01           # [d_model]

print(f"\n步骤 1: 第一层线性投影 (扩展维度)")
print(f"  输入形状: {input_embeddings.shape} [seq_len={seq_len}, d_model={d_model}]")
print(f"  权重形状: {linear1_weight.shape} [d_ff={d_ff}, d_model={d_model}]")

# Linear1: 投影到更高维度
linear1_output = torch.matmul(input_embeddings, linear1_weight.T) + linear1_bias
print(f"  输出形状: {linear1_output.shape} [seq_len={seq_len}, d_ff={d_ff}]")
print(f"  第1个词的第一层输出 (前5个值): {linear1_output[0, :5].tolist()}")

print(f"\n步骤 2: 应用 ReLU 激活函数")
print(f"  ReLU 的作用: max(0, x) - 引入非线性，让网络能学习复杂关系")
relu_output = F.relu(linear1_output)
print(f"  输出形状: {relu_output.shape}")
print(f"  第1个词经过 ReLU 后 (前5个值): {relu_output[0, :5].tolist()}")
print(f"  对比: 有负数变0, 正数保持不变")

print(f"\n步骤 3: 第二层线性投影 (投影回原维度)")
print(f"  输入形状: {relu_output.shape} [seq_len={seq_len}, d_ff={d_ff}]")
print(f"  权重形状: {linear2_weight.shape} [d_model={d_model}, d_ff={d_ff}]")

linear2_output = torch.matmul(relu_output, linear2_weight.T) + linear2_bias
print(f"  输出形状: {linear2_output.shape} [seq_len={seq_len}, d_model={d_model}]")
print(f"  第1个词的最终输出: {linear2_output[0].tolist()}")

print(f"\n总结: FFN 完整过程")
print(f"  输入  [seq_len={seq_len}, d_model={d_model}]")
print(f"    ↓ Linear1 (扩展到 d_ff={d_ff})")
print(f"  中间  [seq_len={seq_len}, d_ff={d_ff}]")
print(f"    ↓ ReLU (非线性激活)")
print(f"  中间  [seq_len={seq_len}, d_ff={d_ff}]")
print(f"    ↓ Linear2 (投影回 d_model={d_model})")
print(f"  输出  [seq_len={seq_len}, d_model={d_model}]")

# ============================================================================
# 第5步：使用 PyTorch 的 FFN 模块
# ============================================================================

print(f"\n【第5步】使用 PyTorch 的 FFN 模块（实际应用）")
print("-" * 100)

class FeedForwardNetwork(nn.Module):
    """标准的前馈网络"""

    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

# 创建 FFN 模块
ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff, dropout=0.0)

# 重新形成输入（加上 batch 维度）
input_with_batch = input_embeddings.unsqueeze(0)  # [1, seq_len, d_model]
print(f"输入形状 (加上 batch 维度): {input_with_batch.shape} [batch=1, seq_len={seq_len}, d_model={d_model}]")

ffn_output = ffn(input_with_batch)
print(f"输出形状: {ffn_output.shape}")
print(f"\n第1个词的输出: {ffn_output[0, 0].tolist()}")

# ============================================================================
# 第6步：理解 FFN 在 Transformer 中的作用
# ============================================================================

print(f"\n【第6步】FFN 在 Transformer 中的作用")
print("-" * 100)

print(f"""
Transformer Encoder 的一层结构:

输入 x
  ↓
【Multi-Head Self-Attention】
  ↓
【Residual Connection】(x + attention_output)
  ↓
【Layer Normalization】
  ↓
【Feed Forward Network】← 我们现在讲的就是这个！
  ↓
【Residual Connection】(x + ffn_output)
  ↓
【Layer Normalization】
  ↓
输出

FFN 的重要性:
  1. 增加模型的非线性表达能力
  2. 每个 token 独立通过 FFN（没有 token 之间的交互）
  3. 与 Attention 互补：
     - Attention: 处理 token 之间的关系
     - FFN: 处理单个 token 内部的特征变换
  4. 计算量大（通常是整个模型的大部分计算）
""")

# ============================================================================
# 第7步：可视化 FFN 对特征的变换
# ============================================================================

print(f"\n【第7步】可视化 FFN 对特征的变换")
print("-" * 100)

print(f"\n追踪第1个词 '我' 的特征变换过程:\n")

# 手动逐步计算
x = input_embeddings[0]  # 第1个词的向量 [8]
print(f"初始向量: {x.tolist()}")
print(f"向量范数: {torch.norm(x).item():.4f}\n")

# 第一层线性
with torch.no_grad():
    h = F.linear(x, ffn.linear1.weight, ffn.linear1.bias)  # [32]
    print(f"经过第一层线性后 (扩展到 32 维):")
    print(f"  向量范数: {torch.norm(h).item():.4f}")
    print(f"  前5个值: {h[:5].tolist()}")
    print(f"  后5个值: {h[-5:].tolist()}\n")

    # ReLU
    h_relu = F.relu(h)
    print(f"经过 ReLU 激活后:")
    print(f"  向量范数: {torch.norm(h_relu).item():.4f}")
    print(f"  前5个值: {h_relu[:5].tolist()}")
    print(f"  后5个值: {h_relu[-5:].tolist()}")
    print(f"  有多少个 0 值: {(h_relu == 0).sum().item()}/{len(h_relu)}\n")

    # 第二层线性
    output = F.linear(h_relu, ffn.linear2.weight, ffn.linear2.bias)  # [8]
    print(f"经过第二层线性后 (投影回 8 维):")
    print(f"  向量范数: {torch.norm(output).item():.4f}")
    print(f"  向量值: {output.tolist()}\n")

print(f"总结: FFN 的作用")
print(f"  输入维度 8 → 扩展到 32 → 压缩回 8")
print(f"  通过扩展和 ReLU，增加了特征的表达能力")
print(f"  这样每个词的特征被深度变换，输出的向量与输入完全不同")

# ============================================================================
# 第8步：对比不同 FFN 配置的影响
# ============================================================================

print(f"\n【第8步】对比不同 FFN 配置的影响")
print("-" * 100)

configs = [
    ("小型 FFN", 8, 16),
    ("标准 FFN (4x)", 8, 32),
    ("大型 FFN (8x)", 8, 64),
]

print(f"\n对同一个输入，用不同大小的 FFN 处理:\n")

for name, d_m, d_f in configs:
    ffn_config = FeedForwardNetwork(d_model=d_m, d_ff=d_f, dropout=0.0)
    test_input = input_with_batch[:, :1, :]  # 只看第1个词

    with torch.no_grad():
        output = ffn_config(test_input)

    # 计算参数量
    params = sum(p.numel() for p in ffn_config.parameters())
    print(f"{name}:")
    print(f"  扩展倍数: {d_f / d_m}x")
    print(f"  参数量: {params}")
    print(f"  计算复杂度: O({d_m} * {d_f} * {d_m}) = {d_m * d_f * d_m}")
    print(f"  输出第一个词的向量范数: {torch.norm(output[0, 0]).item():.4f}\n")

print(f"\n观察:")
print(f"  - FFN 越大，参数越多，模型容量越大")
print(f"  - 但也会导致计算量和过拟合风险增加")
print(f"  - 标准配置是 4x (d_ff = 4 * d_model)，在性能和效率之间找到平衡")

# ============================================================================
# 第9步：完整的 Encoder Layer 演示
# ============================================================================

print(f"\n【第9步】FFN 在完整 Encoder Layer 中的作用")
print("-" * 100)

class EncoderLayer(nn.Module):
    """Transformer 编码器的一个层"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Args:
            x: [seq_len, batch_size, d_model]
        """
        # Multi-Head Attention + Residual + LayerNorm
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        # FFN + Residual + LayerNorm
        ffn_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x

encoder_layer = EncoderLayer(d_model=d_model, num_heads=2, d_ff=d_ff, dropout=0.0)

# 准备输入 (encoder 需要 [seq_len, batch_size, d_model])
encoder_input = input_embeddings.unsqueeze(1)  # [seq_len, batch_size=1, d_model]
print(f"Encoder 输入形状: {encoder_input.shape}")

with torch.no_grad():
    encoder_output = encoder_layer(encoder_input)

print(f"Encoder 输出形状: {encoder_output.shape}")
print(f"\n第1个词的最终输出: {encoder_output[0, 0].tolist()}")

print(f"\n关键点:")
print(f"  1. 输入经过 Attention（理解上下文关系）")
print(f"  2. 加入残差连接（保留原始信息）")
print(f"  3. LayerNorm（归一化特征）")
print(f"  4. 经过 FFN（变换特征空间）")
print(f"  5. 再次残差连接和 LayerNorm")
print(f"  → 最终输出既保留原始信息，又学到了深层特征")

# ============================================================================
# 第10步：总结
# ============================================================================

print(f"\n【第10步】总结")
print("-" * 100)

print(f"""
FFN 的核心原理 (重要!!!):

1. 架构:
   FFN(x) = Linear_2(ReLU(Linear_1(x)))

2. 作用:
   • 增加非线性: ReLU 引入非线性，让模型能学习复杂关系
   • 增加容量: 扩展到 d_ff 维度后再压缩，能学习更丰富的特征组合
   • 独立处理: 每个 token 的处理是独立的，没有 token 之间的交互

3. 计算流程:
   输入 [seq_len, d_model]
     ↓ Linear1: d_model → d_ff (通常 4x 扩展)
   [seq_len, d_ff]
     ↓ ReLU (非线性)
   [seq_len, d_ff]
     ↓ Linear2: d_ff → d_model (投影回)
   输出 [seq_len, d_model]

4. 在 Transformer 中的位置:
   与 Attention 互补
   • Attention: 处理 token 之间的依赖关系
   • FFN: 处理单个 token 内部的特征变换

5. 为什么是 4x?
   • 标准配置: d_ff = 4 * d_model
   • 经验值，在性能和效率之间找到平衡
   • 有时也会用 8x 或其他倍数

6. 实际作用:
   • 占 Transformer 计算量的大部分（通常 60-70%）
   • 但参数量也很大
   • 对模型的理解能力至关重要
""")

print("\n" + "=" * 100)
print("演示完成！")
print("=" * 100 + "\n")
