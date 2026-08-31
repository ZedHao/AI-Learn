# 手搓 Transformer 完整指南

这是一个**从零开始手工实现的 Transformer 架构**，包含详细讲解、代码、例子和测试。

## 📁 文件结构

| 文件 | 说明 |
|------|------|
| `transformer_from_scratch.py` | **核心实现** - 完整的 Transformer 模型及所有组件 |
| `transformer_example.py` | **使用示例** - 包括数据集、训练、推理、生成 |
| `transformer_explanation.py` | **深度讲解** - 13 个方面的详细原理讲解 |
| `test_transformer.py` | **单元测试** - 验证所有组件的正确性 |

## 🚀 快速开始

### 1. 运行测试（验证实现）

```bash
python test_transformer.py
```

输出：✓ 所有测试通过！

### 2. 运行完整例子（训练和推理）

```bash
python transformer_example.py
```

这将：
- 创建随机数据集
- 初始化 Transformer 模型
- 训练 3 个 epoch
- 进行推理演示（贪心搜索和 Beam Search）

### 3. 阅读讲解文档

```bash
python transformer_explanation.py
```

会显示 13 个主题的详细讲解。

## 🏗️ 核心组件详解

### 1. 位置编码（Positional Encoding）

```python
pe = PositionalEncoding(d_model=512, max_seq_length=5000)
x_with_pos = pe(x)  # 添加位置信息
```

**为什么需要？**
- Attention 是置换不变的（不关心顺序）
- 需要显式添加位置信息

**如何工作？**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

---

### 2. 缩放点积注意力（Scaled Dot-Product Attention）

```python
attention = ScaledDotProductAttention(d_k=64)
output, weights = attention(query, key, value)
```

**公式：**
```
Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V
```

**三个关键步骤：**
1. 计算相似度：`scores = Q·K^T`
2. 缩放：`scores = scores / sqrt(d_k)`
3. 加权求和：`output = softmax(scores) · V`

---

### 3. 多头注意力（Multi-Head Attention）

```python
mha = MultiHeadAttention(d_model=512, num_heads=8)
output, weights = mha(query, key, value)
```

**好处：**
- 每个 head 关注不同类型的信息
- 增加表现力，无需增加计算复杂度
- 提高模型能力

**工作流程：**
```
多个 Q, K, V 投影 -> 计算 Attention -> 拼接所有 heads -> 输出投影
```

---

### 4. 前馈网络（Feed Forward Network）

```python
ffn = FeedForwardNetwork(d_model=512, d_ff=2048)
output = ffn(x)
```

**结构：**
```
[d_model] --ReLU--> [d_ff] --Linear--> [d_model]
```

**作用：**
- 引入非线性变换
- Transformer 参数的大部分都在 FFN 中

---

### 5. 编码器层（Encoder Layer）

```python
encoder_layer = EncoderLayer(d_model=512, num_heads=8, d_ff=2048)
output = encoder_layer(x, padding_mask)
```

**结构：**
```
x -> [Self-Attention] -> [Add & Norm] -> [FFN] -> [Add & Norm] -> output
```

**残差连接 + 层归一化的好处：**
- 梯度流动更好
- 训练更稳定
- 支持更深的网络

---

### 6. 解码器层（Decoder Layer）

```python
decoder_layer = DecoderLayer(d_model=512, num_heads=8, d_ff=2048)
output = decoder_layer(tgt, encoder_output, causal_mask, cross_attention_mask)
```

**结构（3 个子层）：**
```
1. 掩码自注意力   (self-attention with causal mask)
2. 交叉注意力     (cross-attention to encoder)
3. 前馈网络       (feed-forward)
```

**特点：**
- 自回归生成（只能看到已生成的 tokens）
- 交叉注意力链接编码器和解码器

---

### 7. 完整 Transformer 模型

```python
model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048
)

# 训练
logits = model(src, tgt)  # [batch, tgt_len, tgt_vocab]

# 推理
tgt = model.greedy_search(src, max_len=100)
```

---

## 💡 关键概念

### 因果掩码（Causal Mask）

在解码器中，位置 t 只能看到位置 0 到 t：

```python
mask = [[1, 0, 0],     # 位置 0 只看位置 0
        [1, 1, 0],     # 位置 1 看位置 0, 1
        [1, 1, 1]]     # 位置 2 看位置 0, 1, 2
```

### Padding 掩码（Padding Mask）

忽略 padding tokens 的贡献：

```python
sequence = [1, 2, 3, 0, 0]  # 0 是 padding
mask = [1, 1, 1, 0, 0]      # 忽略 padding
```

### 层归一化（Layer Normalization）

```python
LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
```

**为什么用 Layer Norm 而不是 Batch Norm？**
- 与 batch 大小无关
- 在序列模型中更稳定

---

## 📊 训练循环示例

```python
# 初始化
trainer = TransformerTrainer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_layers=6
)

# 训练
for epoch in range(num_epochs):
    train_loss = trainer.train_epoch(train_loader)
    val_loss = trainer.validate(val_loader)
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
```

---

## 🎯 推理和生成

### 贪心搜索

```python
inference = TransformerInference(model)
tgt = inference.greedy_search(src, max_len=50)
```

**特点：**
- 每步选择概率最高的 token
- 快速但质量一般

### Beam Search

```python
tgt = inference.beam_search(src, beam_width=5, max_len=50)
```

**特点：**
- 保持 k 个最有概率的候选序列
- 质量更好但更慢

---

## 📈 模型规模

对于 `d_model=512, num_heads=8, num_layers=6, d_ff=2048` 的模型：

```
编码器参数:     ~3.7M
解码器参数:     ~3.7M
嵌入层:         ~20M (取决于词表大小)
─────────────
总计:           ~27M (对于词表大小 ~50k)
```

标准 Transformer (Base)：110M
标准 Transformer (Big)：370M

---

## 🔍 调试技巧

### 1. 检查梯度流动

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.abs().mean():.6f}")
```

### 2. 检查 Attention 权重

```python
# Attention 权重不应该太集中
print(f"Max attention weight: {attention_weights.max()}")
print(f"Min attention weight: {attention_weights.min()}")
```

### 3. 检查位置编码

```python
pe = PositionalEncoding(d_model=512)
print(pe.pe.shape)  # [1, max_seq_length, d_model]
```

---

## 📚 参考文献

本实现基于以下论文：

1. **"Attention Is All You Need"** (Vaswani et al., 2017)
   - Transformer 原始论文
   - https://arxiv.org/abs/1706.03762

2. **"Improving Language Understanding by Generative Pre-Training"** (Radford et al., 2018)
   - GPT 论文
   
3. **"BERT: Pre-training of Deep Bidirectional Transformers"** (Devlin et al., 2018)
   - BERT 论文

---

## ✅ 测试覆盖

```bash
python test_transformer.py
```

测试内容：
- ✓ 位置编码
- ✓ 缩放点积注意力
- ✓ 多头注意力
- ✓ 前馈网络
- ✓ 编码器层
- ✓ 解码器层
- ✓ 完整 Transformer
- ✓ 梯度流动
- ✓ 不同序列长度

---

## 🎓 学习路径

建议按以下顺序学习：

1. **阅读讲解**（`transformer_explanation.py`）
   - 理解 Attention 机制
   - 理解位置编码
   - 理解 Multi-Head Attention

2. **查看实现**（`transformer_from_scratch.py`）
   - 从简单组件到复杂组件
   - 理解形状变换
   - 理解数据流

3. **运行测试**（`test_transformer.py`）
   - 验证理解
   - 实验参数变化

4. **运行示例**（`transformer_example.py`）
   - 完整训练循环
   - 推理演示

---

## 🤔 常见问题

**Q: 为什么 Attention 是 O(seq_len^2) 的复杂度？**

A: 因为需要计算每个位置对其他所有位置的相似度，形成一个 seq_len × seq_len 的矩阵。

**Q: 为什么需要 Softmax 归一化？**

A: 使得 attention 权重成为有效的概率分布（和为 1），便于梯度流动。

**Q: 为什么除以 sqrt(d_k)？**

A: 防止点积过大导致 softmax 梯度消失。d_k 越大，点积越大，所以需要更大的缩放因子。

**Q: 编码器和解码器有什么区别？**

A: 
- 编码器：可以看到全部序列（全注意力）
- 解码器：只能看到当前及之前的位置（因果掩码）

**Q: 如何处理变长序列？**

A: 
- 使用 padding 到统一长度
- 用 padding mask 忽略填充部分

---

## 📝 代码示例总结

### 最小化例子

```python
import torch
from transformer.transformer_from_scratch import Transformer

# 初始化模型
model = Transformer(
    src_vocab_size=1000,
    tgt_vocab_size=1000,
    d_model=256,
    num_heads=8,
    num_layers=4
)

# 准备数据
src = torch.randint(0, 1000, (4, 20))  # [batch_size, src_seq_len]
tgt = torch.randint(0, 1000, (4, 15))  # [batch_size, tgt_seq_len]

# 前向传播
logits = model(src, tgt)  # [4, 15, 1000]

# 计算损失
import torch.nn.functional as F

loss = F.cross_entropy(
    logits.view(-1, 1000),
    tgt.view(-1)
)
```

---

## 📖 额外资源

- **The Annotated Transformer**: http://nlp.seas.harvard.edu/annotated-transformer/
- **Hugging Face Transformers**: https://huggingface.co/transformers/
- **Jay Alammar 的讲解**: http://jalammar.github.io/illustrated-transformer/

---

**Happy Learning! 🚀**
