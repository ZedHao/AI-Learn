"""
Transformer 使用示例和详细讲解
包含: 模型初始化、前向传播、训练循环、推理等
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformer_from_scratch import Transformer


# ============================================================================
# 1. 简单的数据集示例
# ============================================================================
class SimpleTranslationDataset(Dataset):
    """
    简单的翻译数据集示例。
    在实际应用中，你需要：
    1. 构建词汇表 (vocabulary)
    2. 将文本转换为 token IDs
    3. 对齐源语言和目标语言序列
    """

    def __init__(self, num_samples=100, src_vocab_size=100, tgt_vocab_size=100,
                 max_len=20, pad_idx=0, eos_idx=2):
        """
        Args:
            num_samples: 数据集样本数
            src_vocab_size: 源语言词表大小
            tgt_vocab_size: 目标语言词表大小
            max_len: 最大序列长度
            pad_idx: padding token 的索引
            eos_idx: 序列结束符 (EOS) 的索引
        """
        self.num_samples = num_samples
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        生成随机的源序列和目标序列。
        实际应用中应从文件或数据库读取。
        """
        # 随机生成源序列长度 (1 到 max_len)
        src_len = np.random.randint(1, self.max_len + 1)
        tgt_len = np.random.randint(1, self.max_len + 1)

        # 生成源序列 (token IDs)
        # 避免使用 pad_idx, eos_idx 作为实际内容
        src = torch.randint(3, self.src_vocab_size, (src_len,))
        tgt = torch.randint(3, self.tgt_vocab_size, (tgt_len,))

        # 添加 EOS token
        src = torch.cat([src, torch.tensor([self.eos_idx])])
        tgt = torch.cat([tgt, torch.tensor([self.eos_idx])])

        # Padding 到 max_len
        if len(src) < self.max_len:
            src = torch.cat([src, torch.full((self.max_len - len(src),), self.pad_idx)])
        else:
            src = src[:self.max_len]

        if len(tgt) < self.max_len:
            tgt = torch.cat([tgt, torch.full((self.max_len - len(tgt),), self.pad_idx)])
        else:
            tgt = tgt[:self.max_len]

        return src, tgt


# ============================================================================
# 2. 自定义 collate 函数
# ============================================================================
def collate_fn(batch):
    """
    DataLoader 的 collate 函数。
    将一个 batch 的样本打包成张量。

    Args:
        batch: 一个 batch 的 (src, tgt) 元组列表

    Returns:
        src_batch: [batch_size, seq_len]
        tgt_batch: [batch_size, seq_len]
    """
    src_list, tgt_list = zip(*batch)
    src_batch = torch.stack(src_list)
    tgt_batch = torch.stack(tgt_list)
    return src_batch, tgt_batch


# ============================================================================
# 3. Transformer 训练类
# ============================================================================
class TransformerTrainer:
    """
    封装 Transformer 模型的训练逻辑。
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        device='cpu',
        pad_idx=0
    ):
        """
        Args:
            src_vocab_size: 源语言词表大小
            tgt_vocab_size: 目标语言词表大小
            d_model: 模型维度
            num_heads: 注意力头数
            num_layers: Transformer 层数
            d_ff: 前馈网络隐层维度
            device: 计算设备 ('cpu' 或 'cuda')
            pad_idx: padding token 的索引
        """
        self.device = device

        # 初始化模型
        self.model = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=0.1,
            pad_idx=pad_idx
        ).to(device)

        # 损失函数
        # ignore_index 忽略 padding tokens 的损失
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.98))

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        self.pad_idx = pad_idx

    def train_epoch(self, train_loader):
        """
        训练一个 epoch。

        Args:
            train_loader: 训练数据 DataLoader

        Returns:
            平均损失
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (src, tgt) in enumerate(train_loader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            # 前向传播
            # 模型输入是完整的 src 和 tgt
            # 输出是 tgt_seq_len 长度的 logits
            logits = self.model(src, tgt)

            # 计算损失
            # logits: [batch_size, tgt_seq_len, tgt_vocab_size]
            # tgt: [batch_size, tgt_seq_len]
            # 需要 reshape 以输入到 CrossEntropyLoss
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                tgt.view(-1)
            )

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度剪裁（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 优化器步骤
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self, val_loader):
        """
        在验证集上评估模型。

        Args:
            val_loader: 验证数据 DataLoader

        Returns:
            验证集平均损失
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for src, tgt in val_loader:
                src = src.to(self.device)
                tgt = tgt.to(self.device)

                logits = self.model(src, tgt)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    tgt.view(-1)
                )

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def fit(self, train_loader, val_loader, num_epochs=10):
        """
        训练模型。

        Args:
            train_loader: 训练数据 DataLoader
            val_loader: 验证数据 DataLoader
            num_epochs: 训练轮数
        """
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)

            # 训练阶段
            train_loss = self.train_epoch(train_loader)
            print(f"Train Loss: {train_loss:.4f}")

            # 验证阶段
            val_loss = self.validate(val_loader)
            print(f"Val Loss: {val_loss:.4f}")

            # 学习率调度
            self.scheduler.step(val_loss)


# ============================================================================
# 4. Transformer 推理（生成）
# ============================================================================
class TransformerInference:
    """
    Transformer 模型推理（序列生成）。
    使用贪心搜索或 beam search 生成目标序列。
    """

    def __init__(self, model, device='cpu', pad_idx=0, eos_idx=2, bos_idx=1):
        """
        Args:
            model: 已训练的 Transformer 模型
            device: 计算设备
            pad_idx: padding token 索引
            eos_idx: 序列结束符索引
            bos_idx: 序列开始符索引
        """
        self.model = model
        self.device = device
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx

    def greedy_search(self, src, max_len=50):
        """
        贪心搜索：每步选择概率最高的 token。

        Args:
            src: 源序列 [1, src_seq_len]
            max_len: 最大生成长度

        Returns:
            生成的目标序列
        """
        self.model.eval()

        with torch.no_grad():
            # 初始化目标序列，开始符是 BOS
            tgt = torch.full((1, 1), self.bos_idx, dtype=torch.long, device=self.device)

            for step in range(max_len):
                # 前向传播
                logits = self.model(src, tgt)  # [1, current_len, tgt_vocab_size]

                # 获取最后一个位置的 logits
                next_token_logits = logits[0, -1, :]  # [tgt_vocab_size]

                # 贪心：选择概率最高的 token
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)

                # 添加到目标序列
                tgt = torch.cat([tgt, next_token], dim=1)

                # 如果生成了 EOS token，停止
                if next_token.item() == self.eos_idx:
                    break

        return tgt.squeeze(0)

    def beam_search(self, src, beam_width=3, max_len=50):
        """
        Beam search：保持最有概率的 k 个候选序列。

        Args:
            src: 源序列
            beam_width: beam 的宽度（保留的候选数）
            max_len: 最大生成长度

        Returns:
            生成的目标序列（最高概率的一个）
        """
        self.model.eval()
        batch_size = src.size(0)

        with torch.no_grad():
            # 初始化 beam
            # beam_scores: [batch_size * beam_width]
            # input_ids: [batch_size * beam_width, current_len]
            beam_scores = torch.zeros(batch_size * beam_width, device=self.device)
            beam_scores[beam_width:] = float('-inf')

            tgt = torch.full(
                (batch_size * beam_width, 1),
                self.bos_idx,
                dtype=torch.long,
                device=self.device
            )

            src = src.repeat_interleave(beam_width, dim=0)

            for step in range(max_len):
                logits = self.model(src, tgt)  # [batch_size * beam_width, current_len, vocab_size]
                next_token_logits = logits[:, -1, :]  # [batch_size * beam_width, vocab_size]

                # 计算条件概率
                log_probs = torch.log_softmax(next_token_logits, dim=-1)

                # 广播 beam_scores 到 vocab 维度
                # scores: [batch_size * beam_width, vocab_size]
                scores = beam_scores.unsqueeze(-1) + log_probs

                # 展平并找到 top-k
                scores_flat = scores.view(batch_size, beam_width, -1).view(batch_size, -1)
                top_scores, top_indices = torch.topk(scores_flat, beam_width, dim=-1)

                # 更新 beam scores
                beam_scores = top_scores.view(-1)

                # 获取对应的 token 和 beam 索引
                beam_indices = top_indices // next_token_logits.size(-1)
                token_indices = top_indices % next_token_logits.size(-1)

                # 重排历史序列和 source
                tgt = tgt[beam_indices + torch.arange(batch_size, device=self.device).unsqueeze(-1) * beam_width]
                src = src[beam_indices + torch.arange(batch_size, device=self.device).unsqueeze(-1) * beam_width]

                # 添加新 token
                next_tokens = token_indices.unsqueeze(-1)
                tgt = torch.cat([tgt, next_tokens], dim=1)

        # 返回概率最高的序列（第一个）
        return tgt[0]


# ============================================================================
# 5. 完整示例
# ============================================================================
def main():
    """
    完整的训练和推理示例。
    """
    print("=" * 80)
    print("Transformer 完整示例")
    print("=" * 80)

    # 配置参数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")

    src_vocab_size = 100
    tgt_vocab_size = 100
    batch_size = 16
    num_epochs = 3
    max_seq_length = 20

    # 1. 准备数据
    print("\n[1] 准备数据...")
    train_dataset = SimpleTranslationDataset(
        num_samples=200,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_len=max_seq_length
    )
    val_dataset = SimpleTranslationDataset(
        num_samples=50,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_len=max_seq_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False
    )

    print(f"  训练集大小: {len(train_dataset)}")
    print(f"  验证集大小: {len(val_dataset)}")
    print(f"  Batch 大小: {batch_size}")

    # 2. 初始化模型
    print("\n[2] 初始化 Transformer 模型...")
    trainer = TransformerTrainer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        device=device
    )

    # 计算模型参数数量
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"  总参数数: {total_params:,}")
    print(f"  可训练参数数: {trainable_params:,}")

    # 3. 训练模型
    print("\n[3] 开始训练...")
    trainer.fit(train_loader, val_loader, num_epochs=num_epochs)

    # 4. 推理
    print("\n[4] 推理示例...")
    inference = TransformerInference(trainer.model, device=device)

    # 创建随机源序列作为示例
    src_example = torch.randint(3, src_vocab_size, (1, max_seq_length))
    src_example = torch.cat(
        [src_example, torch.full((1, max_seq_length - src_example.size(1)), 0)],
        dim=1
    )[:, :max_seq_length].to(device)

    print(f"\n  源序列 (token IDs): {src_example[0][:10].tolist()}... (显示前 10 个)")

    # 贪心搜索
    print("\n  [4.1] 使用贪心搜索生成目标序列...")
    tgt_greedy = inference.greedy_search(src_example, max_len=max_seq_length)
    print(f"  生成的目标序列: {tgt_greedy.cpu().tolist()}")

    # Beam search
    print("\n  [4.2] 使用 Beam Search (beam_width=3) 生成目标序列...")
    tgt_beam = inference.beam_search(src_example, beam_width=3, max_len=max_seq_length)
    print(f"  生成的目标序列: {tgt_beam.cpu().tolist()}")

    print("\n" + "=" * 80)
    print("示例完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
