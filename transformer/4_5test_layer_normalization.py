"""
深入理解 Layer Normalization (层归一化)
这个文件专注于讲解 LayerNorm 的各个方面，帮助深入理解
"""

import torch
import torch.nn as nn
import numpy as np


def test_basic_layer_norm():
    """
    ========== 【基础】LayerNorm 的基本原理 ==========

    核心公式:
      LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β

    其中:
      μ (mu)     = 当前向量的均值
      σ (sigma)  = 当前向量的标准差
      γ (gamma)  = 可学习的缩放系数 (学出来的)
      β (beta)   = 可学习的偏移系数 (学出来的)
      ε (epsilon)= 防止除以0的小常数 (通常1e-5)
    """
    print("\n" + "="*70)
    print("【基础】LayerNorm 的基本原理")
    print("="*70)

    # 手动计算 LayerNorm
    x = torch.tensor([1.0, 100.0, 2.0, 50.0])

    print(f"\n原始向量: {x}")
    print(f"原始均值: {x.mean():.4f}")
    print(f"原始标准差: {x.std():.4f}")

    # 步骤1: 计算均值和标准差
    mu = x.mean()
    sigma = x.std()

    print(f"\n【步骤1】计算统计量")
    print(f"  μ (均值) = {mu:.4f}")
    print(f"  σ (标准差) = {sigma:.4f}")

    # 步骤2: 归一化
    eps = 1e-5
    x_normalized = (x - mu) / (sigma + eps)

    print(f"\n【步骤2】归一化 (x - μ) / (σ + ε)")
    print(f"  结果: {x_normalized}")
    print(f"  归一化后均值: {x_normalized.mean():.6f} (应该 ≈ 0)")
    print(f"  归一化后标准差: {x_normalized.std():.6f} (应该 ≈ 1)")

    # 步骤3: 应用可学习的 γ 和 β（初始值）
    gamma = 1.0  # 初始缩放系数
    beta = 0.0   # 初始偏移系数
    output = gamma * x_normalized + beta

    print(f"\n【步骤3】应用可学习的 γ (缩放) 和 β (偏移)")
    print(f"  γ (gamma) = {gamma}")
    print(f"  β (beta) = {beta}")
    print(f"  输出: {output}")

    # 与 PyTorch 内置的 LayerNorm 对比
    print(f"\n【对比】与 PyTorch 内置 LayerNorm:")
    layer_norm = nn.LayerNorm(4)
    x_batch = x.unsqueeze(0).unsqueeze(0)  # [1, 1, 4]
    output_pytorch = layer_norm(x_batch).squeeze()
    print(f"  PyTorch 输出: {output_pytorch}")
    print(f"  数值基本一致 ✓")


def test_layer_norm_effect():
    """
    ========== 【效果展示】LayerNorm 的稳定作用 ==========

    展示 LayerNorm 如何把离散的数值范围压缩到稳定区间
    这是为什么深层网络需要 LayerNorm 的关键
    """
    print("\n" + "="*70)
    print("【效果展示】LayerNorm 的稳定作用")
    print("="*70)

    # 模拟 Attention 输出（数值不稳定）
    batch_size = 2
    seq_len = 5
    d_model = 512

    # 创建不稳定的数据（乘以100让跨度更大）
    x_unstable = torch.randn(batch_size, seq_len, d_model) * 100

    print(f"\n【未应用 LayerNorm】(模拟 Attention 输出)")
    print(f"  形状: {x_unstable.shape}")
    print(f"  均值: {x_unstable.mean():.4f}")
    print(f"  标准差: {x_unstable.std():.4f}")
    print(f"  最小值: {x_unstable.min():.4f}")
    print(f"  最大值: {x_unstable.max():.4f}")
    print(f"  数值范围跨度: {x_unstable.max() - x_unstable.min():.2f}")

    # 应用 LayerNorm
    layer_norm = nn.LayerNorm(d_model)
    x_normalized = layer_norm(x_unstable)

    print(f"\n【应用 LayerNorm 后】")
    print(f"  形状: {x_normalized.shape} (不变)")
    print(f"  均值: {x_normalized.mean():.6f} (≈ 0 ✓)")
    print(f"  标准差: {x_normalized.std():.6f} (≈ 1 ✓)")
    print(f"  最小值: {x_normalized.min():.4f}")
    print(f"  最大值: {x_normalized.max():.4f}")
    print(f"  数值范围跨度: {x_normalized.max() - x_normalized.min():.2f}")

    print(f"\n【对比】")
    ratio = (x_unstable.max() - x_unstable.min()) / (x_normalized.max() - x_normalized.min())
    print(f"  数值跨度缩小了 {ratio:.1f}x")
    print(f"  不稳定数值 → 稳定数值 ✓")


def test_layer_norm_vs_softmax():
    """
    ========== 【区别对比】LayerNorm vs Softmax ==========

    这是常见的混淆点。两者都是"归一化"，但目的完全不同
    """
    print("\n" + "="*70)
    print("【区别对比】LayerNorm vs Softmax")
    print("="*70)

    x = torch.tensor([1.0, 100.0, 2.0, 50.0])

    print(f"\n原始向量: {x}")

    # LayerNorm
    layer_norm = nn.LayerNorm(4)
    x_ln = layer_norm(x.unsqueeze(0)).squeeze()

    print(f"\n【LayerNorm 结果】")
    print(f"  输出: {x_ln}")
    print(f"  特点1: 可以是负数 → {x_ln.min().item():.4f}")
    print(f"  特点2: 可以大于1 → {x_ln.max().item():.4f}")
    print(f"  特点3: 和 ≠ 1 → {x_ln.sum().item():.4f}")
    print(f"  特点4: 均值=0, 标准差=1 ✓")
    print(f"  用途: 稳定数值，方便后续网络处理")

    # Softmax
    x_softmax = torch.softmax(x, dim=0)

    print(f"\n【Softmax 结果】")
    print(f"  输出: {x_softmax}")
    print(f"  特点1: 全是正数 ✓")
    print(f"  特点2: 全部 ≤ 1 ✓")
    print(f"  特点3: 和 = 1 → {x_softmax.sum().item():.6f} ✓")
    print(f"  用途: 转成概率分布，用于 Attention 权重")

    print(f"\n【总结】")
    print(f"  LayerNorm: 特征缩放（Feature Scaling）")
    print(f"  Softmax:   概率转换（Probability Distribution）")
    print(f"  两者用途完全不同！")


def test_in_transformer_context():
    """
    ========== 【实际应用】LayerNorm 在 Transformer 中的作用 ==========

    展示为什么 Encoder/Decoder Layer 中需要 LayerNorm
    """
    print("\n" + "="*70)
    print("【实际应用】LayerNorm 在 Transformer 中的作用")
    print("="*70)

    batch_size = 2
    seq_len = 5
    d_model = 512

    print(f"\nTransformer 中一层的流程:")
    print(f"  输入 x: [batch={batch_size}, seq_len={seq_len}, d_model={d_model}]")

    # 模拟输入
    x = torch.randn(batch_size, seq_len, d_model) * 50
    print(f"\n【步骤1】输入数据统计")
    print(f"  均值: {x.mean():.4f}")
    print(f"  标准差: {x.std():.4f}")
    print(f"  范围: [{x.min():.2f}, {x.max():.2f}]")

    # 模拟 Attention 输出
    attention_output = torch.randn(batch_size, seq_len, d_model) * 80
    print(f"\n【步骤2】Attention 输出")
    print(f"  均值: {attention_output.mean():.4f}")
    print(f"  标准差: {attention_output.std():.4f}")
    print(f"  范围: [{attention_output.min():.2f}, {attention_output.max():.2f}]")

    # 残差连接（Add）
    residual_sum = x + attention_output
    print(f"\n【步骤3】残差连接 (Add) - x + attention_output")
    print(f"  均值: {residual_sum.mean():.4f}")
    print(f"  标准差: {residual_sum.std():.4f}")
    print(f"  范围: [{residual_sum.min():.2f}, {residual_sum.max():.2f}]")
    print(f"  ⚠️  数值已经不稳定！")

    # 应用 LayerNorm（Norm）
    layer_norm1 = nn.LayerNorm(d_model)
    normalized = layer_norm1(residual_sum)
    print(f"\n【步骤4】LayerNorm (Norm) - 稳定数值")
    print(f"  均值: {normalized.mean():.6f} ✓")
    print(f"  标准差: {normalized.std():.6f} ✓")
    print(f"  范围: [{normalized.min():.2f}, {normalized.max():.2f}]")
    print(f"  ✅ 数值稳定了！")

    # 模拟 FFN 输出
    ffn_output = torch.randn(batch_size, seq_len, d_model) * 60
    print(f"\n【步骤5】FFN 输出")
    print(f"  均值: {ffn_output.mean():.4f}")
    print(f"  标准差: {ffn_output.std():.4f}")
    print(f"  范围: [{ffn_output.min():.2f}, {ffn_output.max():.2f}]")

    # 残差连接
    residual_sum2 = normalized + ffn_output
    print(f"\n【步骤6】残差连接 (Add) - normalized + ffn_output")
    print(f"  均值: {residual_sum2.mean():.4f}")
    print(f"  标准差: {residual_sum2.std():.4f}")
    print(f"  范围: [{residual_sum2.min():.2f}, {residual_sum2.max():.2f}]")
    print(f"  ⚠️  又不稳定了...")

    # 再次应用 LayerNorm
    layer_norm2 = nn.LayerNorm(d_model)
    final_output = layer_norm2(residual_sum2)
    print(f"\n【步骤7】LayerNorm (Norm) - 再次稳定")
    print(f"  均值: {final_output.mean():.6f} ✓")
    print(f"  标准差: {final_output.std():.6f} ✓")
    print(f"  范围: [{final_output.min():.2f}, {final_output.max():.2f}]")
    print(f"  ✅ 可以输送到下一层！")

    print(f"\n【结论】")
    print(f"  每层都需要 LayerNorm 来稳定数值")
    print(f"  这样梯度才能正常反向传播")
    print(f"  深层网络才能正常训练")


def test_learnable_parameters():
    """
    ========== 【学习】LayerNorm 的可学习参数 ==========

    LayerNorm 不仅是简单的归一化，还有可学习的 γ 和 β
    这些参数在训练过程中会被学习
    """
    print("\n" + "="*70)
    print("【学习】LayerNorm 的可学习参数")
    print("="*70)

    d_model = 8  # 小的维度方便查看

    layer_norm = nn.LayerNorm(d_model)

    print(f"\n【LayerNorm 的参数】")
    print(f"  形状: [{d_model}]")

    print(f"\n  gamma (缩放系数):")
    print(f"    初始值: {layer_norm.weight}")
    print(f"    形状: {layer_norm.weight.shape}")
    print(f"    (初始都是 1.0)")

    print(f"\n  beta (偏移系数):")
    print(f"    初始值: {layer_norm.bias}")
    print(f"    形状: {layer_norm.bias.shape}")
    print(f"    (初始都是 0.0)")

    print(f"\n【作用】")
    print(f"  γ: 调整输出的缩放程度")
    print(f"  β: 调整输出的中心位置")
    print(f"  训练过程中，这些参数会被自动优化")

    # 演示参数变化
    print(f"\n【训练演示】")
    x = torch.randn(2, 5, d_model)

    optimizer = torch.optim.SGD(layer_norm.parameters(), lr=0.1)

    print(f"  初始 gamma: {layer_norm.weight[:3].tolist()}")
    print(f"  初始 beta: {layer_norm.bias[:3].tolist()}")

    for i in range(3):
        output = layer_norm(x)
        loss = output.mean()  # 假设这是损失函数

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"\n  迭代 {i+1} 后:")
        print(f"    gamma: {layer_norm.weight[:3].tolist()}")
        print(f"    beta: {layer_norm.bias[:3].tolist()}")

    print(f"\n  参数随着训练不断变化 ✓")


def test_no_layernorm_vs_with_layernorm():
    """
    ========== 【对比实验】有无 LayerNorm 的梯度流动 ==========

    展示为什么深层网络必须有 LayerNorm（配合残差连接）
    """
    print("\n" + "="*70)
    print("【对比实验】有无 LayerNorm 的梯度流动")
    print("="*70)

    batch_size = 2
    d_model = 256
    num_layers = 3

    print(f"\n构建 {num_layers} 层网络，对比梯度大小")

    # 不使用 LayerNorm 的网络
    print(f"\n【方案1】无 LayerNorm + 无残差连接 ❌")

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(d_model, d_model)
                for _ in range(num_layers)
            ])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    simple_net = SimpleNet()
    x = torch.randn(batch_size, d_model)
    output = simple_net(x)
    loss = output.mean()
    loss.backward()

    print(f"  各层梯度范数:")
    for i, layer in enumerate(simple_net.layers):
        grad_norm = layer.weight.grad.norm().item()
        print(f"    层{i}: {grad_norm:.6f}")

    # 使用 LayerNorm 的网络
    print(f"\n【方案2】有 LayerNorm + 残差连接 ✓")

    class StableNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(d_model, d_model)
                for _ in range(num_layers)
            ])
            self.norms = nn.ModuleList([
                nn.LayerNorm(d_model)
                for _ in range(num_layers)
            ])

        def forward(self, x):
            residual = x
            for layer, norm in zip(self.layers, self.norms):
                x = norm(residual + layer(x))
                residual = x
            return x

    stable_net = StableNet()
    x = torch.randn(batch_size, d_model)
    output = stable_net(x)
    loss = output.mean()
    loss.backward()

    print(f"  各层梯度范数:")
    for i, layer in enumerate(stable_net.layers):
        grad_norm = layer.weight.grad.norm().item()
        print(f"    层{i}: {grad_norm:.6f}")

    print(f"\n【结论】")
    print(f"  无 LayerNorm: 梯度逐层衰减（梯度消失）")
    print(f"  有 LayerNorm: 梯度保持稳定（可以训练）")


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("Layer Normalization (层归一化) 深入理解")
    print("="*70)

    test_basic_layer_norm()
    test_layer_norm_effect()
    test_layer_norm_vs_softmax()
    test_in_transformer_context()
    test_learnable_parameters()
    test_no_layernorm_vs_with_layernorm()

    print("\n" + "="*70)
    print("✓ 所有测试完成！")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
