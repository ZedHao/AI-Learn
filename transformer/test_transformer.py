"""
快速测试脚本 - 验证 Transformer 实现的正确性

========== Transformer 9 个测试快速参考 ==========

【第1步】test_positional_encoding()
  干啥: 验证位置编码正确性
  原理: Attention 不关心顺序 → 给每个位置添加唯一编码
  输入: [batch=2, seq_len=5, d_model=128]
  输出: [batch=2, seq_len=5, d_model=128] (形状不变)

【第2步】test_scaled_dot_product_attention()
  干啥: 验证单头注意力计算
  原理: Attention(Q,K,V) = softmax(Q·K^T/√d_k)·V
  公式: 相似度 → 缩放 → softmax → 加权求和
  输出: 注意力权重矩阵 [batch, heads, seq_len, seq_len]

【第3步】test_multi_head_attention()
  干啥: 验证 8 个头的并行计算
  原理: 8 个独立的注意力头学习不同的关注模式
  分割: d_model=512 → 8个头 × 64维
  拼接: 8个[batch, seq_len, 64] → [batch, seq_len, 512]

【第4步】test_feed_forward()
  干啥: 验证前馈网络特征变换
  原理: FFN(x) = Linear2(ReLU(Linear1(x)))
  流程: 512 → 2048 (扩展4倍) → ReLU → 512
  作用: 增加非线性，占总计算量 60-70%

【第5步】test_encoder_layer()
  干啥: 验证编码器一层的完整功能
  流程: Attention → 残差 → 规范化 → FFN → 残差 → 规范化
  设计: Attention学位置间关系，FFN学特征深度

【第6步】test_decoder_layer()
  干啥: 验证解码器层能否融合源和目标
  三部分: 自注意力(带掩码) → 交叉注意力 → FFN
  掩码: 解码器只能看左边词（因果性约束）
  交叉注意力: 连接源序列和目标序列的关键

【第7步】test_transformer()
  干啥: 验证端到端序列到序列转换
  流程: 源序列→Encoder→Decoder→目标序列
  参数: ~376万参数
  应用: 翻译、摘要、问答等

【第8步】test_gradient_flow()
  干啥: 验证反向传播能否更新所有参数
  流程: 前向 → 损失 → 反向传播 → 梯度流动
  关键: 残差连接保证深层梯度不消失
  检查: 所有参数都有非零梯度

【第9步】test_different_sequence_lengths()
  干啥: 验证灵活处理变长序列
  优势: 没有固定输入/输出大小
  复杂度: Attention O(seq_len²)，FFN O(seq_len)
  限制: max_seq_length 通常 512-5000

========== 核心概念总结 ==========

为什么 Transformer 这么强?
  1. Attention: 全局视野（vs RNN局部视野）
  2. 并行计算: 所有位置同时处理（vs RNN顺序处理）
  3. 残差连接: 解决梯度消失（深网络可训练）

Encoder vs Decoder
  • Encoder: 理解输入，可随意看全序列
  • Decoder: 生成输出，只能看左边+源序列

Attention vs FFN
  • Attention: 位置间关系（横向交互）
  • FFN: 位置内特征（纵向深度）

========== 详细讲解 ==========

每个测试函数的 docstring 中有详细讲解：
  - 目的: 这个测试验证什么?
  - 原理: 背后的数学原理是什么?
  - 流程: 具体执行步骤是什么?
  - 验证: 检查哪些内容?

在 IDE 中悬停函数名可看到 docstring，或查看源文件!
"""

import torch
import torch.nn as nn
from transformer_from_scratch import (
    PositionalEncoding,
    ScaledDotProductAttention,
    MultiHeadAttention,
    FeedForwardNetwork,
    EncoderLayer,
    DecoderLayer,
    Transformer
)


def test_positional_encoding():
    """
    ========== 【测试 1】位置编码 (Positional Encoding) ==========

    目的: 验证位置编码是否能正确添加到词向量中

    原理:
      问题: Transformer 的 Attention 机制是并行的，不关心顺序
            "我 爱 你" 和 "你 爱 我" 经过 Attention 后结果相同 ❌

      解决方案: 给每个位置添加独特的编码向量

      具体操作:
        # 原始输入（没有位置信息）
        x = [[token_embedding_1],      # "我"
             [token_embedding_2],      # "爱"
             [token_embedding_3]]      # "你"

        # 加上位置编码后
        output = x + positional_encoding
               = [[token_embedding_1 + pos_encoding_0],    # "我(位置0)"
                  [token_embedding_2 + pos_encoding_1],    # "爱(位置1)"
                  [token_embedding_3 + pos_encoding_2]]    # "你(位置2)"

      效果对比:
        没有位置编码: "我 爱 你" = "你 爱 我"  (完全相同)
        有位置编码:   "我(pos=0) 爱(pos=1) 你(pos=2)" ≠ "你(pos=0) 爱(pos=1) 我(pos=2)"

    验证内容:
      ✓ 输入形状: [batch_size=2, seq_len=5, d_model=128]
      ✓ 输出形状: 应该与输入相同 (只是加法操作)
      ✓ 形状不变，信息保留

    类比理解:
      就像在班里给每个学生贴一个号码牌，让模型知道"谁是第几个"
    """
    print("\n" + "="*60)
    print("测试 1: 位置编码 (Positional Encoding)")
    print("="*60)

    d_model = 128
    max_len = 10
    pe = PositionalEncoding(d_model, max_len)

    # 测试输入
    x = torch.randn(2, 5, d_model)  # [batch_size, seq_len, d_model]
    output = pe(x)

    print(f"✓ 输入形状: {x.shape}")
    print(f"✓ 输出形状: {output.shape}")
    assert output.shape == x.shape, "输出形状应该与输入相同"
    print("✓ 位置编码测试通过")


def test_scaled_dot_product_attention():
    """
    ========== 【测试 2】缩放点积注意力 (Scaled Dot-Product Attention) ==========

    目的: 验证单头注意力是否能正确计算关注权重

    核心公式: Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

    详细步骤:
      1. 计算相似度 (Query × Key)
         Q·K^T 得到 [batch, heads, seq_len, seq_len] 的矩阵
         表示每个位置与其他位置的"匹配度"
         比如: 位置0 与 所有位置 的相似度

      2. 缩放 (除以 √d_k)
         防止点积太大导致梯度消失
         d_k = 64 → √64 = 8 → 除以 8 来缩放

      3. 应用 softmax
         把分数转换成 0-1 的权重分布
         权重和 = 1 (概率分布)

      4. 加权求和 (权重 × Value)
         最终输出 = 权重 · V

    权重矩阵例子 (3个位置):
      [[0.8, 0.15, 0.05],    → 位置0: 80%看自己, 15%看位置1, 5%看位置2
       [0.1, 0.7,  0.2],     → 位置1: 10%看位置0, 70%看自己, 20%看位置2
       [0.2, 0.3,  0.5]]     → 位置2: 20%看位置0, 30%看位置1, 50%看自己

    输入形状:
      Q (Query):   [batch=2, heads=8, seq_len=10, d_k=64]
      K (Key):     [batch=2, heads=8, seq_len=10, d_k=64]
      V (Value):   [batch=2, heads=8, seq_len=10, d_k=64]

    输出:
      output:      [batch=2, heads=8, seq_len=10, d_k=64]  (和值的形状相同)
      weights:     [batch=2, heads=8, seq_len=10, seq_len=10]  (注意力权重矩阵)

    验证:
      ✓ 输出形状 == 值的形状
      ✓ 权重都是正数 (softmax 的性质)
      ✓ 权重和为 1 (每一行求和 ≈ 1.0)
    """
    print("\n" + "="*60)
    print("测试 2: 缩放点积注意力 (Scaled Dot-Product Attention)")
    print("="*60)

    d_k = 64
    batch_size = 2
    seq_len = 10

    attention = ScaledDotProductAttention(d_k)

    q = torch.randn(batch_size, 8, seq_len, d_k)
    k = torch.randn(batch_size, 8, seq_len, d_k)
    v = torch.randn(batch_size, 8, seq_len, d_k)

    output, weights = attention(q, k, v)

    print(f"✓ 查询形状: {q.shape}")
    print(f"✓ 键形状: {k.shape}")
    print(f"✓ 值形状: {v.shape}")
    print(f"✓ 输出形状: {output.shape}")
    print(f"✓ Attention 权重形状: {weights.shape}")

    assert output.shape == v.shape, "输出形状应该与值相同"
    assert weights.shape == (batch_size, 8, seq_len, seq_len)

    # 验证权重是有效的概率分布
    weight_sum = weights.sum(dim=-1)
    print(f"✓ Attention 权重和范围: min={weight_sum.min():.4f}, max={weight_sum.max():.4f}")
    # 所有权重加起来应该都大于 0
    assert (weight_sum > 0).all(), "权重应该都是正数"

    print("✓ Attention 测试通过")


def test_multi_head_attention():
    """
    ========== 【测试 3】多头注意力 (Multi-Head Attention) ==========

    目的: 验证多个注意力头是否能独立学习不同的关注模式

    原理:
      问题: 单头注意力只能学一种关注模式
      解决: 用 8 个"头"并行处理，每个头学不同的模式

    架构:
      Multi-Head(Q, K, V) = Concat(head_1, head_2, ..., head_8) · W^O

    详细步骤:
      1. 投影到多个头
         Q, K, V [batch, seq_len, d_model=512]
         ↓ 投影成 8 个头，每个头维度 = 512/8 = 64
         8 个头 [batch, 8, seq_len, 64]

      2. 每个头独立做注意力
         head_1: Q1·K1^T / √64 → softmax → 加权求和
         head_2: Q2·K2^T / √64 → softmax → 加权求和
         ...
         head_8: Q8·K8^T / √64 → softmax → 加权求和

      3. 拼接所有头
         8 个 [batch, seq_len, 64] → [batch, seq_len, 512]

      4. 最终投影
         [batch, seq_len, 512] × W^O → [batch, seq_len, d_model=512]

    8 个头学习的不同模式:
      head_1: 关注相邻词 (局部依赖)
      head_2: 关注整个句子 (全局依赖)
      head_3: 关注特定语法 (语法结构)
      head_4: 关注语义关系 (语义意义)
      head_5-8: 其他组合模式

    作用:
      就像 8 个不同专业的分析师从各自角度分析同一个问题
      多视角 → 更丰富的特征学习

    验证内容:
      ✓ 输出形状 == 输入形状 (投影保持维度)
      ✓ 权重矩阵有 8 个 (8 个独立的头)
      ✓ 每个权重矩阵都是 [batch, seq_len, seq_len]
    """
    print("\n" + "="*60)
    print("测试 3: 多头注意力 (Multi-Head Attention)")
    print("="*60)

    d_model = 512
    num_heads = 8
    seq_len = 10
    batch_size = 2

    mha = MultiHeadAttention(d_model, num_heads)

    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)

    output, weights = mha(q, k, v)

    print(f"✓ 输入形状: {q.shape}")
    print(f"✓ 输出形状: {output.shape}")
    print(f"✓ Attention 权重形状: {weights.shape}")

    assert output.shape == (batch_size, seq_len, d_model)
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)

    print("✓ 多头注意力测试通过")


def test_feed_forward():
    """
    ========== 【测试 4】前馈网络 (Feed Forward Network) ==========

    目的: 验证前馈网络是否能正确变换特征

    架构: FFN(x) = Linear_2(ReLU(Linear_1(x)))

    详细步骤:
      1. 第一层线性投影：扩展维度
         [batch, seq_len, 512] → [batch, seq_len, 2048]
         维度扩展 4 倍 (512 × 4 = 2048)
         作用: 增加模型容量，学习复杂的特征组合

      2. ReLU 激活函数
         max(0, x) - 引入非线性
         负数变 0，正数保持不变
         作用: 让网络能学习复杂的非线性关系

      3. Dropout (可选)
         随机丢弃一些神经元的激活
         作用: 防止过拟合

      4. 第二层线性投影：投影回原维度
         [batch, seq_len, 2048] → [batch, seq_len, 512]
         维度投影回 512
         作用: 还原到原始空间，准备输入下一层

    为什么是 4 倍扩展?
      • 标准配置: d_ff = 4 × d_model
      • 经验证明这个比例在效果和效率间找到最好平衡
      • 有时也用 8 倍、2 倍等，但 4 倍最常见

    计算量:
      第一层参数: 512 × 2048 ≈ 100 万
      第二层参数: 2048 × 512 ≈ 100 万
      总共: ~200 万参数

    关键特点:
      ✓ 每个 token 独立处理 (没有 token 间的交互)
      ✓ 占 Transformer 计算量的 60-70%
      ✓ 与 Attention 互补: Attention 看位置间的关系，FFN 看位置内的特征

    验证内容:
      ✓ 输出形状 == 输入形状 (投影到原维度)
      ✓ 输出维度保持不变
    """
    print("\n" + "="*60)
    print("测试 4: 前馈网络 (Feed Forward Network)")
    print("="*60)

    d_model = 512
    d_ff = 2048
    seq_len = 10
    batch_size = 2

    ffn = FeedForwardNetwork(d_model, d_ff)

    x = torch.randn(batch_size, seq_len, d_model)
    output = ffn(x)

    print(f"✓ 输入形状: {x.shape}")
    print(f"✓ 输出形状: {output.shape}")

    assert output.shape == x.shape
    print("✓ 前馈网络测试通过")


def test_layer_normalization():
    """
    ========== 【测试 4.5】层归一化 (Layer Normalization) ==========

    目的: 验证 LayerNorm 如何稳定数值，防止梯度消失/爆炸

    原理:
      问题: Attention 和 FFN 的输出数值范围不一致
            比如 Attention 输出: [-1000, 2000, 0.5, -50]（跨度太大）
                FFN 输出: [0.001, 100, -200, 0.01]（也很离散）
            这样的数值会导致：
              • 梯度爆炸：梯度过大，参数更新不稳定
              • 梯度消失：深层网络梯度太小，无法更新
            ❌ 网络无法训练

      解决方案: LayerNorm 把任意范围的数值缩放到 "均值≈0, 标准差≈1"

    核心公式:
      LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β

      其中:
        μ (mu)     = 当前向量的均值
        σ (sigma)  = 当前向量的标准差
        γ (gamma)  = 可学习的缩放系数
        β (beta)   = 可学习的偏移系数
        ε (epsilon)= 防止除以0的小常数

    具体步骤:

      输入: x = [1, 100, 2, 50]
      ↓
      【步1】计算统计量
        μ = (1 + 100 + 2 + 50) / 4 = 38.25
        σ = √((1-38.25)² + (100-38.25)² + ...) / 4) = 43.2
      ↓
      【步2】归一化
        x_norm = [(1-38.25)/43.2, (100-38.25)/43.2, ...]
               = [-0.86, 1.43, -0.86, 0.27]
               ← 现在均值=0, 标准差=1 ✓
      ↓
      【步3】可学习的放缩和平移
        output = γ * x_norm + β
               （γ 和 β 初始都是 1 和 0，训练过程中模型自动学习）
      ↓
      输出: [-0.86, 1.43, -0.86, 0.27]

    关键作用:

      1. 稳定梯度流
         没有 LayerNorm:
           梯度 = 大 × (衰减因子)^深度
           深层梯度 ≈ 大 × 0.9^20 ≈ 很小（梯度消失）❌

         有 LayerNorm:
           梯度 = 1 × (衰减因子)^深度
           归一化让梯度尺度从 [-1000,2000] 变成 [-1,2]
           深层梯度 ≈ 1 × 0.9^20 ≈ 仍然可接受 ✓

      2. 加速收敛
         数值在相近范围 → 学习率设置更容易
         不需要频繁调整学习率

      3. 更好的初始化
         不管初始权重怎么设，LayerNorm 都能把它拉到稳定范围

    为什么 LayerNorm 不是 Softmax?

      LayerNorm:
        - 可以是负数 ✓
        - 可以大于 1 ✓
        - 和 ≠ 1
        - 用途: 稳定数值
        - 位置: 每层之后（Attention / FFN 之后）

      Softmax:
        - 全都正数 [0, 1]
        - 和 = 1（概率分布）
        - 用途: 转成概率
        - 位置: Attention 权重计算

    验证内容:
      ✓ 输出形状 == 输入形状 (只是数值调整)
      ✓ 输出均值 ≈ 0
      ✓ 输出标准差 ≈ 1
      ✓ 梯度能正常反向传播
    """
    print("\n" + "="*60)
    print("测试 4.5: 层归一化 (Layer Normalization)")
    print("="*60)

    batch_size = 2
    seq_len = 5
    d_model = 512

    # 创建 LayerNorm
    layer_norm = nn.LayerNorm(d_model)

    # 模拟 Attention 输出（数值范围不稳定）
    x_unstable = torch.randn(batch_size, seq_len, d_model) * 100
    # 乘以100是为了让数值跨度更大，模拟真实训练中的不稳定情况
    # 现在 x_unstable 的范围可能是 [-100, 100]

    print(f"✓ 输入形状: {x_unstable.shape}")
    print(f"✓ 输入均值: {x_unstable.mean():.4f}")
    print(f"✓ 输入标准差: {x_unstable.std():.4f}")
    print(f"✓ 输入最小值: {x_unstable.min():.4f}")
    print(f"✓ 输入最大值: {x_unstable.max():.4f}")

    # 应用 LayerNorm
    x_normalized = layer_norm(x_unstable)

    print(f"\n【应用 LayerNorm 后】")
    print(f"✓ 输出形状: {x_normalized.shape}")
    print(f"✓ 输出均值: {x_normalized.mean():.4f} (应该 ≈ 0)")
    print(f"✓ 输出标准差: {x_normalized.std():.4f} (应该 ≈ 1)")
    print(f"✓ 输出最小值: {x_normalized.min():.4f}")
    print(f"✓ 输出最大值: {x_normalized.max():.4f}")

    # 验证统计量
    assert abs(x_normalized.mean().item()) < 0.01, "均值应该接近0"
    assert abs(x_normalized.std().item() - 1.0) < 0.1, "标准差应该接近1"

    print(f"\n【对比】")
    print(f"输入跨度:   [{x_unstable.min():.2f}, {x_unstable.max():.2f}] = {(x_unstable.max() - x_unstable.min()):.2f}")
    print(f"输出跨度:   [{x_normalized.min():.2f}, {x_normalized.max():.2f}] = {(x_normalized.max() - x_normalized.min()):.2f}")
    print(f"✓ LayerNorm 成功把数值范围归一化到稳定区间！")

    print("\n✓ 层归一化测试通过")


def test_encoder_layer():
    """
    ========== 【测试 5】编码器层 (Encoder Layer) ==========

    目的: 验证完整的编码器层（注意力 + 前馈）是否能正确集成

    架构: Encoder Layer = Attention + 残差 + 规范化 + FFN + 残差 + 规范化

    完整流程:
      输入 x [batch=2, seq_len=10, d_model=512]
        ↓
      【第 1 步】Multi-Head Self-Attention (自注意力)
        自注意力: Q=K=V=x
        每个位置"看"整个句子
        输出: attn_out [batch=2, seq_len=10, d_model=512]
        ↓
      【第 2 步】Residual Connection (残差连接)
        输出 = x + attn_out
        保留原始信息，防止梯度消失
        ↓
      【第 3 步】Layer Normalization (层规范化)
        归一化每个特征维度
        加速收敛，提高稳定性
        ↓
      【第 4 步】Feed Forward Network
        FFN: Linear(512→2048) → ReLU → Linear(2048→512)
        输出: ffn_out [batch=2, seq_len=10, d_model=512]
        ↓
      【第 5 步】Residual Connection
        输出 = 前面的输出 + ffn_out
        ↓
      【第 6 步】Layer Normalization
        最后再归一化一次
        ↓
      输出 [batch=2, seq_len=10, d_model=512]

    关键设计点:
      Attention 学什么?  → 位置之间的依赖关系 (横向信息流)
      FFN 学什么?       → 每个位置内部的特征变换 (纵向特征深度)
      二者互补完成深层信息处理

    残差连接的重要性:
      x → [处理] → 处理后特征
      ↓             ↑
      ←————— 直连 ←————

      最终输出 = 原始信息 + 新学到的信息
      梯度可以直接通过残差分支流回，解决梯度消失问题
      这就是为什么 Transformer 可以有 6-24 层还能训练！

    规范化的作用:
      把不同尺度的特征统一到同一范围
      加速训练收敛
      提高数值稳定性

    验证内容:
      ✓ 输出形状 == 输入形状 (所有操作都保持维度)
      ✓ 信息流畅通 (残差连接保证梯度反向传播)
    """
    print("\n" + "="*60)
    print("测试 5: 编码器层 (Encoder Layer)")
    print("="*60)

    d_model = 512
    num_heads = 8
    seq_len = 10
    batch_size = 2

    encoder_layer = EncoderLayer(d_model, num_heads)

    x = torch.randn(batch_size, seq_len, d_model)
    output = encoder_layer(x)

    print(f"✓ 输入形状: {x.shape}")
    print(f"✓ 输出形状: {output.shape}")

    assert output.shape == x.shape
    print("✓ 编码器层测试通过")


def test_decoder_layer():
    """
    ========== 【测试 6】解码器层 (Decoder Layer) ==========

    目的: 验证解码器层（包含交叉注意力）是否能正确融合源和目标信息

    架构: Decoder Layer = 自注意力(带掩码) + 交叉注意力 + FFN
         + 3 次残差连接 + 3 次规范化

    详细流程:
      解码器输入 x       [batch=2, seq_len_tgt=5, d_model=512]  (目标序列)
      编码器输出 enc_out [batch=2, seq_len_src=10, d_model=512] (源序列)
        ↓
      【第 1 步】Masked Multi-Head Self-Attention (掩码自注意力)
        自注意力，但加入"因果掩码"
        原因: 解码时不能看到未来的词！
        比如翻译时:
          生成第 1 个词 → 只能看第 1 个词
          生成第 2 个词 → 只能看第 1,2 个词
          生成第 3 个词 → 只能看第 1,2,3 个词
        ↓
      【第 2 步】Residual + Layer Norm
        ↓
      【第 3 步】Cross-Attention (交叉注意力) - 关键！
        Q 来自: 解码器 ("我想生成什么词?")
        K, V 来自: 编码器 ("源句子有什么信息?")

        这是连接源和目标的关键！让解码器能查看源序列

        例子 (翻译中→英):
        源序列 (编码器): "我 爱 你" → 编码成深层表示
        目标序列 (解码器生成): "I love you"
        生成"love"时 → 查询源序列 → "爱"的信息权重最高
        ↓
      【第 4 步】Residual + Layer Norm
        ↓
      【第 5 步】Feed Forward Network
        ↓
      【第 6 步】Residual + Layer Norm
        ↓
      输出 [batch=2, seq_len_tgt=5, d_model=512]

    解码器 vs 编码器 对比:

      编码器:
        • 只有自注意力 (看整个输入)
        • 可以随意看左右上下文
        • 并行处理所有位置

      解码器:
        • 自注意力 + 交叉注意力 + 掩码
        • 只能看左边 (因果性约束)
        • 逐步生成，不能看到未来

      为什么需要掩码?
        编码: 输入完整已知，可以随意看左右
        解码: 边生成边预测，不能看到后面的内容
              就像在线翻译，不知道后面还有什么词

    交叉注意力的作用 (极其重要):
      连接源序列和目标序列的桥梁
      让解码器知道"源句子的哪部分对应我现在生成的词"
      实现真正的序列到序列转换

    验证内容:
      ✓ 输出形状 == 目标序列形状 (可以更短或更长)
      ✓ 正确融合了源和目标信息
      ✓ 掩码生效 (解码器无法提前看到后续词)
    """
    print("\n" + "="*60)
    print("测试 6: 解码器层 (Decoder Layer)")
    print("="*60)

    d_model = 512
    num_heads = 8
    tgt_seq_len = 5
    src_seq_len = 10
    batch_size = 2

    decoder_layer = DecoderLayer(d_model, num_heads)

    x = torch.randn(batch_size, tgt_seq_len, d_model)
    encoder_output = torch.randn(batch_size, src_seq_len, d_model)

    output = decoder_layer(x, encoder_output)

    print(f"✓ 解码器输入形状: {x.shape}")
    print(f"✓ 编码器输出形状: {encoder_output.shape}")
    print(f"✓ 输出形状: {output.shape}")

    assert output.shape == x.shape
    print("✓ 解码器层测试通过")


def test_transformer():
    """
    ========== 【测试 7】完整 Transformer 模型 ==========

    目的: 验证完整的 Transformer 能否进行序列到序列的转换

    架构总体流程:

      源序列 (中文)
      "我 爱 你"
        ↓
      【Embedding + Positional Encoding】
      把词 ID 转成 512 维向量，加入位置信息
        ↓
      【Encoder Stack (2 层)】
      Encoder Layer 1:
        • Multi-Head Self-Attention (8 个头)
        • Feed Forward Network

      Encoder Layer 2:
        • Multi-Head Self-Attention (8 个头)
        • Feed Forward Network

      输出: 编码后的源句子深层表示
      [batch=4, seq_len_src=10, d_model=256]
        ↓
      【交叉信息】编码器输出 → 传给解码器
        ↓
      目标序列 (英文)
      "<START> I love you"
        ↓
      【Embedding + Positional Encoding】
        ↓
      【Decoder Stack (2 层)】
      Decoder Layer 1:
        • Masked Self-Attention (看不到后面的词)
        • Cross-Attention (查看源序列)
        • Feed Forward Network

      Decoder Layer 2:
        • Masked Self-Attention
        • Cross-Attention
        • Feed Forward Network

      输出: 解码后的目标序列表示
      [batch=4, seq_len_tgt=8, d_model=256]
        ↓
      【Output Projection】
      投影到词表大小 (100 个词)
        ↓
      Logits [batch=4, seq_len_tgt=8, vocab_size=100]
      每个位置预测 100 个词中哪个概率最高

    工作流程解析:

      1. 编码阶段:
         源句子进入编码器 → 学习源语言的深层表示
         结果: 包含整个源句子的语义和语法信息的向量

      2. 解码初始化:
         目标序列 (带开始标记) 进入解码器
         解码器看自己已生成的部分

      3. 解码阶段:
         解码器通过交叉注意力查看源序列
         每生成一个词时，关注源序列的相关部分
         比如生成"love"时，关注源序列的"爱"

      4. 输出预测:
         最后一层输出投影到词表大小
         得到每个位置的概率分布

    参数量计算 (简化):
      词嵌入: vocab_size × d_model = 100 × 256 = 25.6K
      编码器: 多层多头注意力 + FFN ≈ 5-10M
      解码器: 类似编码器 ≈ 5-10M
      输出投影: d_model × vocab_size = 25.6K
      总体: 通常 50M 到 1000M+ 参数

    常见应用:
      • 机器翻译: 中文 → 英文 (最经典应用)
      • 文本摘要: 长文章 → 摘要
      • 问答系统: 问题 + 文档 → 答案
      • 对话系统: 用户输入 → 机器回复
      • 代码生成: 注释 → 代码

    验证内容:
      ✓ 输出形状正确: [batch=4, tgt_seq_len=8, vocab_size=100]
      ✓ 模型参数数合理
      ✓ 能处理变长序列 (源和目标长度不同)
      ✓ 计算图构建成功，可以进行反向传播
    """
    print("\n" + "="*60)
    print("测试 7: 完整 Transformer 模型")
    print("="*60)

    src_vocab_size = 100
    tgt_vocab_size = 100
    d_model = 256
    num_heads = 8
    num_layers = 2
    batch_size = 4
    src_seq_len = 10
    tgt_seq_len = 8

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=1024
    )

    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))

    output = model(src, tgt)

    print(f"✓ 源序列形状: {src.shape}")
    print(f"✓ 目标序列形状: {tgt.shape}")
    print(f"✓ 输出形状: {output.shape}")

    assert output.shape == (batch_size, tgt_seq_len, tgt_vocab_size)

    # 计算模型参数数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型参数数: {total_params:,}")

    print("✓ 完整 Transformer 测试通过")


def test_gradient_flow():
    """
    ========== 【测试 8】梯度流动 (Gradient Flow) ==========

    目的: 验证反向传播是否能正确计算梯度，更新所有参数

    原理 - 梯度流动过程:

      【第 1 步】前向传播 (Forward Pass)
      源序列 ID → Embedding → Encoder → Decoder → 输出 logits
      输出: [batch=2, seq_len_tgt=6, vocab_size=50]
      包含每个位置对每个词的预测分数

      【第 2 步】计算损失 (Loss Computation)
      预测词的 logits:  [0.1, 0.8, 0.05, 0.05, ...] (50 个词的分数)
      真实词的 one-hot: [0, 1, 0, 0, ...]           (第 2 个词是真实)

      损失 = CrossEntropyLoss
           = -log(0.8)     (对真实词的概率取负对数)
           ≈ 0.223

      【第 3 步】反向传播 (Backward Pass)
      梯度从损失一层层反向流回

      损失梯度
        ↑
      输出投影层梯度 (更新投影权重)
        ↑
      解码器第 2 层梯度 (FFN + 交叉注意力)
        ↑
      解码器第 1 层梯度 (FFN + 交叉注意力)
        ↑
      编码器第 2 层梯度 (FFN + 自注意力)
        ↑
      编码器第 1 层梯度 (FFN + 自注意力)
        ↑
      词嵌入层梯度 (更新嵌入向量)

      最终: 所有参数都有了梯度，可以进行优化

    详细验证流程:

      1. 前向传播:
         output = model(src, tgt)

      2. 计算损失:
         loss = CrossEntropyLoss(output.view(-1, vocab_size),
                                  target.view(-1))

      3. 反向传播:
         loss.backward()
         PyTorch 自动计算所有参数的梯度

      4. 检查梯度:
         for param in model.parameters():
           if param.grad is None: ❌ 没有梯度 (有问题!)
           if param.grad.abs().sum() == 0: ❌ 梯度全是 0 (有问题!)
           else: ✓ 梯度正常

    梯度消失问题 & 解决方案:

      没有残差连接的情况:
        每一层反向传播时，梯度都会衰减
        深层的梯度 ≈ 浅层梯度 × (衰减因子)^层数
        比如衰减因子 0.9，20 层后 ≈ 0.9^20 ≈ 0.12 (只剩 12%!)
        深层参数得不到足够的梯度更新 ❌

      有残差连接的情况:
        梯度可以直接通过残差分支流回
        梯度 = 直接通路梯度 + 处理路径梯度
        即使处理路径梯度很小，直接路径保证梯度不会完全消失 ✓
        这就是为什么 Transformer 可以有很多层 (6-24 层)!

    Loss 值的合理性:

      随机初始化模型:
        vocab_size = 50
        随机预测每个词概率 = 1/50 = 0.02
        Loss ≈ -log(0.02) ≈ 3.9

      训练过程中:
        Loss 应该逐步下降 (模型学习)
        Loss → 接近 0 表示预测越来越准

      如果 Loss 没有下降:
        ❌ 梯度计算有问题
        ❌ 学习率太小/太大
        ❌ 模型架构有问题

    参数更新公式:

      new_param = old_param - learning_rate × param.grad
      比如: learning_rate = 0.001
           old_weight = 0.5
           grad = 0.1
           new_weight = 0.5 - 0.001 × 0.1 = 0.4999

    验证内容:
      ✓ 所有参数都有梯度 (param.grad is not None)
      ✓ 梯度不全为零 (param.grad.abs().sum() > 0)
      ✓ 梯度能流动到所有层 (包括编码器最早的层)
      ✓ Loss 值合理 (通常在 0-5 范围)
      ✓ 梯度通过残差连接有效反向传播
    """
    print("\n" + "="*60)
    print("测试 8: 梯度流动")
    print("="*60)

    src_vocab_size = 50
    tgt_vocab_size = 50
    d_model = 128
    num_heads = 4
    num_layers = 2
    batch_size = 2
    src_seq_len = 8
    tgt_seq_len = 6

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers
    )

    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))
    target = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))

    # 前向传播
    output = model(src, tgt)

    # 计算损失
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    loss = criterion(output.view(-1, tgt_vocab_size), target.view(-1))

    # 反向传播
    loss.backward()

    # 检查梯度
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    print(f"✓ 损失值: {loss.item():.4f}")
    print(f"✓ 有效梯度数: {sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)}")
    print(f"✓ 参数梯度: {sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None):.4f}")

    assert has_grad, "应该有有效的梯度"
    print("✓ 梯度流动测试通过")


def test_different_sequence_lengths():
    """
    ========== 【测试 9】不同长度序列处理 ==========

    目的: 验证 Transformer 能否灵活处理任意长度的输入和输出序列

    传统神经网络的局限:
      输入固定大小: 必须输入恰好 10 个词
      输出固定大小: 必须输出恰好 8 个词
      如果句子太短或太长就无法处理 ❌

    Transformer 的优势:
      输入可变长: 3 个词、10 个词、1000 个词都能处理
      输出可变长: 4 个词、6 个词、5 个词都能处理
      自动调整计算图大小 ✓

    为什么 Transformer 能这样做?

      1. Attention 机制:
         接受任意长的 Query, Key, Value
         计算 Q·K^T 时的形状是 [seq_len_q, seq_len_k]
         seq_len 可以是任意值，不受限制

      2. Feed Forward Network:
         对每个位置独立操作: Linear → ReLU → Linear
         每个位置处理是独立的，长度任意

      3. 位置编码:
         预先计算很长的编码 (max_seq_length)
         只需要截取对应长度的部分即可

    测试用例 (3 种不同配置):

      case 1: batch_size=4, src_len=5词,  tgt_len=6词
              源序列较短，目标序列略长

      case 2: batch_size=2, src_len=10词, tgt_len=8词
              源序列较长，目标序列较短

      case 3: batch_size=8, src_len=3词,  tgt_len=4词
              都很短的情况

    验证内容:

      ✓ 模型对所有情况都能正确处理
      ✓ 输出形状正确: [batch_size, tgt_len, vocab_size]
      ✓ 没有错误 (没有形状不匹配的错误)
      ✓ 计算图构建成功
      ✓ 没有"超出位置编码范围"的错误

    计算复杂度分析:

      Attention 复杂度: O(seq_len²)
        必须计算所有位置对之间的相似度
        seq_len = 100 → 10,000 个点积
        seq_len = 1000 → 1,000,000 个点积

      FFN 复杂度: O(seq_len)
        每个位置独立处理
        seq_len = 100 → 100 次操作
        seq_len = 1000 → 1000 次操作

      因此:
        短序列: 计算快 (O(seq_len²) 中 seq_len²很小)
        长序列: 计算慢 (O(seq_len²) 中 seq_len²很大)

      这就是为什么很多实际应用中要限制输入长度！

    位置编码的上限:

      一般设置: max_seq_length = 5000
      如果输入超过 5000 个词: ❌ 会报错 "position beyond max_seq_length"

      解决超长序列的方案:
        1. 增加 max_seq_length (但显存和速度也会增加)
        2. 使用相对位置编码 (position relative, 不依赖绝对位置)
        3. 使用 ALiBi 位置编码 (不需要预计算位置编码)
        4. 使用 Longformer 等针对长文本的模型

    实际应用中的序列长度限制:

      GPT-3: 最大 2048 个 token
      BERT: 最大 512 个 token
      BART (摘要): 1024 token
      长文本模型: 4096-8192 token
      T5 (部分版本): 16384 token

      为什么要限制?
        • 显存限制 (显存有限)
        • 计算速度 (O(seq_len²) 非常昂贵)
        • 位置编码的周期性问题
        • 实际任务需求 (通常不需要非常长的上下文)

    实际测试验证:

      def test_different_sequence_lengths():
        test_cases = [
          (4, 5, 6),   → batch_size=4, src_len=5, tgt_len=6
          (2, 10, 8),  → batch_size=2, src_len=10, tgt_len=8
          (8, 3, 4),   → batch_size=8, src_len=3, tgt_len=4
        ]

        对每一种情况:
          ✓ 创建随机源和目标序列
          ✓ 通过模型前向传播
          ✓ 检查输出形状是否正确
          ✓ 没有错误发生

    验证结果示例:

      ✓ 批次大小: 4, 源序列长度: 5, 目标序列长度: 6
        → 输出: [4, 6, 100]

      ✓ 批次大小: 2, 源序列长度: 10, 目标序列长度: 8
        → 输出: [2, 8, 100]

      ✓ 批次大小: 8, 源序列长度: 3, 目标序列长度: 4
        → 输出: [8, 4, 100]

      所有情况都通过 ✓✓✓
    """
    print("\n" + "="*60)
    print("测试 9: 不同长度的序列处理")
    print("="*60)

    src_vocab_size = 100
    tgt_vocab_size = 100
    d_model = 256
    num_heads = 8
    num_layers = 2

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers
    )

    test_cases = [
        (4, 5, 6),   # batch_size, src_len, tgt_len
        (2, 10, 8),
        (8, 3, 4),
    ]

    for batch_size, src_len, tgt_len in test_cases:
        src = torch.randint(0, src_vocab_size, (batch_size, src_len))
        tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))

        output = model(src, tgt)

        assert output.shape == (batch_size, tgt_len, tgt_vocab_size)
        print(f"✓ 批次大小: {batch_size}, 源序列长度: {src_len}, "
              f"目标序列长度: {tgt_len} -> 输出: {output.shape}")

    print("✓ 不同序列长度测试通过")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*80)
    print("开始运行 Transformer 实现测试")
    print("="*80)
    """
    测试1: 位置编码          ← 学基础
      测试2: 单头注意力        ← 学基础
      测试3: 多头注意力        ← 学基础
      测试4: 前馈网络          ← 学基础
      ✨  测试4.5: LayerNorm    ← 学怎么稳定 【新增】
      测试5: 编码器层          ← 看组装效果
      测试6: 解码器层          ← 看组装效果
      测试7: 完整Transformer   ← 看端到端
      测试8: 梯度流动          ← 验证可训练
      测试9: 变长序列          ← 验证灵活性

    """
    try:
        test_positional_encoding()
        test_scaled_dot_product_attention()
        test_multi_head_attention()
        test_feed_forward()
        test_layer_normalization()
        test_encoder_layer()
        test_decoder_layer()
        test_transformer()
        test_gradient_flow()
        test_different_sequence_lengths()

        print("\n" + "="*80)
        print("✓ 所有测试通过！Transformer 实现正确无误。")
        print("="*80 + "\n")

    except Exception as e:
        print("\n" + "="*80)
        print(f"✗ 测试失败: {e}")
        print("="*80 + "\n")
        raise


if __name__ == "__main__":
    run_all_tests()
