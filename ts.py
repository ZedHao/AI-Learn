import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------------------- 1. 位置编码 ----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0)]
        return x

# ---------------------- 2. 多头自注意力 ----------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(1)

        # 线性映射 + 分头
        q = self.w_q(q).view(-1, batch_size, self.nhead, self.d_k).transpose(1,2)
        k = self.w_k(k).view(-1, batch_size, self.nhead, self.d_k).transpose(1,2)
        v = self.w_v(v).view(-1, batch_size, self.nhead, self.d_k).transpose(1,2)

        # 缩放点积注意力
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, -1e9)
        attn_weight = F.softmax(attn_score, dim=-1)

        # 加权求和
        output = torch.matmul(attn_weight, v)

        # 拼接多头
        output = output.transpose(1,2).contiguous().view(-1, batch_size, self.d_model)
        output = self.out(output)
        return output

# ---------------------- 3. FeedForward 网络 ----------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# ---------------------- 4. Encoder 层 ----------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, nhead)
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # 自注意力 + 残差
        attn_out = self.attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        # FFN + 残差
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x

# ---------------------- 5. Decoder 层 ----------------------
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.cross_attn = MultiHeadAttention(d_model, nhead)
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None):
        # 解码器自注意力
        attn1 = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(attn1))

        # 交叉注意力（Encoder输出作为KV）
        attn2 = self.cross_attn(tgt, memory, memory)
        tgt = self.norm2(tgt + self.dropout2(attn2))

        # FFN
        ffn_out = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout3(ffn_out))
        return tgt

# ---------------------- 6. 完整 Transformer ----------------------
class HandCraftTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        # 嵌入 + 位置编码
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)

        # 堆叠 Encoder/Decoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

        # 输出映射
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_square_subsequent_mask(self, sz):
        """生成Decoder掩码，屏蔽未来位置"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        # src/tgt: [seq_len, batch_size]
        src_seq_len, _ = src.shape
        tgt_seq_len, _ = tgt.shape

        # 嵌入 + 位置编码
        src_emb = self.dropout(self.pe(self.src_emb(src) * math.sqrt(self.d_model)))
        tgt_emb = self.dropout(self.pe(self.tgt_emb(tgt) * math.sqrt(self.d_model)))

        # Encoder 前向
        memory = src_emb
        for enc_layer in self.encoder_layers:
            memory = enc_layer(memory)

        # Decoder 掩码
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)

        # Decoder 前向
        output = tgt_emb
        for dec_layer in self.decoder_layers:
            output = dec_layer(output, memory, tgt_mask)

        # 输出词汇概率
        output = self.fc_out(output)
        return output

# ---------------------- 测试运行 ----------------------
if __name__ == "__main__":
    # 模拟参数
    src_vocab = 5000
    tgt_vocab = 5000
    batch_size = 2
    src_len = 10
    tgt_len = 12

    # 随机模拟输入
    src = torch.randint(0, src_vocab, (src_len, batch_size))
    tgt = torch.randint(0, tgt_vocab, (tgt_len, batch_size))

    # 初始化手搓Transformer
    model = HandCraftTransformer(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        d_model=512,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3
    )

    # 前向传播
    out = model(src, tgt)
    print("模型输出形状:", out.shape)  # [tgt_len, batch_size, tgt_vocab_size]