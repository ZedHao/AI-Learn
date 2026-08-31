"""
手搓 Transformer 完整实现 - 从零开始理解 Transformer 架构
主要包含: Attention, MultiHeadAttention, FeedForward, Encoder, Decoder
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ============================================================================
# 1. 位置编码 (Positional Encoding)
# ============================================================================
class PositionalEncoding(nn.Module):
    """
    给序列添加位置信息。由于 Self-Attention 是并行处理的，
    没有先后顺序信息，所以需要显式地添加位置编码来保留序列顺序。

    使用正弦和余弦函数：
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵: [max_seq_length, d_model]
        pe = torch.zeros(max_seq_length, d_model)

        # 位置索引: [max_seq_length, 1]
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        # 分母项: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            -(np.log(10000.0) / d_model)
        )

        # 应用正弦和余弦
        pe[:, 0::2] = torch.sin(position * div_term)        # 偶数位置用 sin
        pe[:, 1::2] = torch.cos(position * div_term)        # 奇数位置用 cos

        # 注册为 buffer，不作为参数更新，但会随 model 移动到对应设备
        self.register_buffer('pe', pe.unsqueeze(0))         # [1, max_seq_length, d_model]

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_length, d_model]
        Returns:
            x + 位置编码
        """
        return self.dropout(x + self.pe[:, :x.size(1), :])


# ============================================================================
# 2. 缩放点积注意力 (Scaled Dot-Product Attention)
# ============================================================================
class ScaledDotProductAttention(nn.Module):
    """
    计算 Attention 权重和输出。

    Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V

    缩放因子 1/sqrt(d_k) 用于防止点积过大导致 softmax 梯度消失。
    """

    def __init__(self, d_k: int, dropout: float = 0.1):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query,
        key,
        value,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, num_heads, seq_len_q, d_k]
            key:   [batch_size, num_heads, seq_len_k, d_k]
            value: [batch_size, num_heads, seq_len_v, d_k]
            mask:  可选，用于 masking（如因果掩码、pad 掩码等）

        Returns:
            output: [batch_size, num_heads, seq_len_q, d_k]
            attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        """

        # 计算点积相似度: [batch_size, num_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.d_k)

        # 应用 mask（可选）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 计算 attention 权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 加权求和
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


# ============================================================================
# 3. 多头注意力 (Multi-Head Attention)
# ============================================================================
class MultiHeadAttention(nn.Module):
    """
    让模型能够关注不同位置的信息，通过多个"头"并行处理。

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O
    其中 head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 投影矩阵
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query,
        key,
        value,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query, key, value: [batch_size, seq_len, d_model]
            mask: 可选掩码

        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)

        # 1. 线性投影并分割成多个头
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, num_heads, d_k]
        # -> [batch_size, num_heads, seq_len, d_k]
        q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. 对每个头计算注意力
        attn_output, attention_weights = self.attention(q, k, v, mask)

        # 3. 拼接所有头: [batch_size, num_heads, seq_len, d_k]
        # -> [batch_size, seq_len, num_heads, d_k] -> [batch_size, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        # 4. 最终线性投影
        output = self.W_o(attn_output)
        output = self.dropout(output)

        return output, attention_weights


# ============================================================================
# 4. 前馈网络 (Feed Forward Network)
# ============================================================================
class FeedForwardNetwork(nn.Module):
    """
    FFN(x) = max(0, x·W_1 + b_1)·W_2 + b_2

    在两个全连接层之间加入 ReLU 激活函数。
    第一层扩展维度（通常是原来的 4 倍），第二层降回原维度。
    """

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


# ============================================================================
# 5. 编码器层 (Encoder Layer)
# ============================================================================
class EncoderLayer(nn.Module):
    """
    Transformer 编码器的一个层。

    结构:
    1. Multi-Head Self-Attention
    2. Add & Norm (残差连接 + Layer Norm)
    3. Feed Forward Network
    4. Add & Norm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: 可选掩码
        Returns:
            output: [batch_size, seq_len, d_model]
        """

        # 残差连接 + Layer Norm: x -> Attention -> Add -> Norm
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 残差连接 + Layer Norm: x -> FFN -> Add -> Norm
        ffn_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x


# ============================================================================
# 6. 解码器层 (Decoder Layer)
# ============================================================================
class DecoderLayer(nn.Module):
    """
    Transformer 解码器的一个层。

    结构:
    1. Masked Multi-Head Self-Attention（只能看到当前及之前的位置）
    2. Add & Norm
    3. Multi-Head Cross-Attention（查询来自解码器，键和值来自编码器）
    4. Add & Norm
    5. Feed Forward Network
    6. Add & Norm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x,
        encoder_output,
        self_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            x: 解码器输入 [batch_size, seq_len_tgt, d_model]
            encoder_output: 编码器输出 [batch_size, seq_len_src, d_model]
            self_attention_mask: 因果掩码（自注意力）
            cross_attention_mask: 交叉注意力掩码

        Returns:
            output: [batch_size, seq_len_tgt, d_model]
        """

        # 1. 掩码自注意力（只能看到当前及之前的位置）
        self_attn_output, _ = self.self_attention(x, x, x, self_attention_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        # 2. 交叉注意力（查询来自解码器，键值来自编码器）
        cross_attn_output, _ = self.cross_attention(
            x, encoder_output, encoder_output, cross_attention_mask
        )
        x = self.norm2(x + self.dropout(cross_attn_output))

        # 3. 前馈网络
        ffn_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ffn_output))

        return x


# ============================================================================
# 7. 编码器 (Encoder)
# ============================================================================
class Encoder(nn.Module):
    """
    多层编码器的堆叠。
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: 可选掩码
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x


# ============================================================================
# 8. 解码器 (Decoder)
# ============================================================================
class Decoder(nn.Module):
    """
    多层解码器的堆叠。
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x,
        encoder_output,
        self_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            x: 解码器输入 [batch_size, seq_len_tgt, d_model]
            encoder_output: 编码器输出 [batch_size, seq_len_src, d_model]
            self_attention_mask: 因果掩码（自注意力）
            cross_attention_mask: 交叉注意力掩码

        Returns:
            output: [batch_size, seq_len_tgt, d_model]
        """
        for layer in self.layers:
            x = layer(x, encoder_output, self_attention_mask, cross_attention_mask)
        return x


# ============================================================================
# 9. 完整 Transformer 模型 (Transformer)
# ============================================================================
class Transformer(nn.Module):
    """
    完整的 Transformer 模型，用于序列到序列任务（如翻译、文本摘要）。

    包含:
    - 词嵌入层
    - 位置编码
    - 编码器堆栈
    - 解码器堆栈
    - 输出投影层
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 5000,
        dropout: float = 0.1,
        pad_idx: int = 0
    ):
        super().__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx

        # 嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

        # 编码器和解码器
        self.encoder = Encoder(d_model, num_layers, num_heads, d_ff, dropout)
        self.decoder = Decoder(d_model, num_layers, num_heads, d_ff, dropout)

        # 输出投影层
        self.projection = nn.Linear(d_model, tgt_vocab_size)

        # 初始化权重
        self._init_parameters()

    def _init_parameters(self):
        """Xavier 初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _create_padding_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """
        创建 padding 掩码。
        pad tokens 被设置为 0，非 pad tokens 被设置为 1。

        返回: [batch_size, 1, 1, seq_len]
        """
        mask = (seq != self.pad_idx).unsqueeze(1).unsqueeze(1)
        return mask

    def _create_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """
        创建因果掩码（自回归掩码）。
        在解码器中，当前位置只能看到当前及之前的位置。

        返回: [1, 1, size, size]
        """
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask == 0
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            src: 源序列 (通常是输入文本 token IDs)
                 [batch_size, src_seq_len]
            tgt: 目标序列 (通常是输出文本 token IDs)
                 [batch_size, tgt_seq_len]

        Returns:
            logits: 输出 token 的 logits
                   [batch_size, tgt_seq_len, tgt_vocab_size]
        """

        device = src.device

        # 1. 编码器部分
        src_embedded = self.src_embedding(src) * np.sqrt(self.d_model)
        src_embedded = self.positional_encoding(src_embedded)

        src_padding_mask = self._create_padding_mask(src)
        encoder_output = self.encoder(src_embedded, src_padding_mask)

        # 2. 解码器部分
        tgt_embedded = self.tgt_embedding(tgt) * np.sqrt(self.d_model)
        tgt_embedded = self.positional_encoding(tgt_embedded)

        tgt_seq_len = tgt.size(1)

        # 创建掩码
        causal_mask = self._create_causal_mask(tgt_seq_len, device)
        tgt_padding_mask = self._create_padding_mask(tgt)

        # 结合因果掩码和 padding 掩码
        tgt_self_attention_mask = causal_mask & tgt_padding_mask

        # 交叉注意力的掩码（source padding）
        cross_attention_mask = src_padding_mask

        # 解码器输出
        decoder_output = self.decoder(
            tgt_embedded,
            encoder_output,
            self_attention_mask=tgt_self_attention_mask,
            cross_attention_mask=cross_attention_mask
        )

        # 3. 投影到词表大小
        logits = self.projection(decoder_output)

        return logits


if __name__ == "__main__":
    print("=" * 80)
    print("手搓 Transformer - 完整实现演示")
    print("=" * 80)
