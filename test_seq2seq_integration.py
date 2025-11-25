"""
测试 Seq2Seq 模型集成

验证修改后的 LSTM/GRU/BiLSTM/BiGRU 模型是否正确输出序列。
"""

import torch
import numpy as np
from models import LSTM, GRU, BiLSTM, BiGRU, ConfigLoader, ModelFactory

def test_baseline_seq2seq():
    """测试基准模型的 Seq2Seq 输出"""
    print("="*70)
    print("测试基准模型的 Seq2Seq 输出")
    print("="*70)

    # 测试参数
    batch_size = 16
    seq_len = 40
    input_size = 16

    # 创建测试数据 (batch, seq_len, input_size)
    x = torch.randn(batch_size, seq_len, input_size)
    print(f"\n输入形状: {x.shape}")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  input_size: {input_size}")

    # 测试所有RNN模型
    models = {
        'LSTM': LSTM(input_size=input_size, hidden_size=64, num_layers=2),
        'GRU': GRU(input_size=input_size, hidden_size=64, num_layers=2),
        'BiLSTM': BiLSTM(input_size=input_size, hidden_size=64, num_layers=2),
        'BiGRU': BiGRU(input_size=input_size, hidden_size=64, num_layers=2),
    }

    for name, model in models.items():
        print(f"\n{'='*70}")
        print(f"测试 {name}")
        print(f"{'='*70}")

        # 前向传播
        model.eval()
        with torch.no_grad():
            output = model(x)

        print(f"输出形状: {output.shape}")
        print(f"  预期: ({batch_size}, {seq_len}, 1)")
        print(f"  实际: {tuple(output.shape)}")

        # 验证输出形状
        assert output.shape == (batch_size, seq_len, 1), \
            f"{name} 输出形状错误！预期 ({batch_size}, {seq_len}, 1)，实际 {output.shape}"

        # 验证输出范围（应该在[0,1]之间，因为有Sigmoid）
        print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        assert output.min() >= 0 and output.max() <= 1, \
            f"{name} 输出范围错误！应在[0,1]之间"

        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"参数量: {total_params:,}")

        print(f"  [OK] {name} 测试通过")

    print("\n" + "="*70)
    print("所有模型测试通过！")
    print("="*70)


def test_model_factory_integration():
    """测试 ModelFactory 集成"""
    print("\n" + "="*70)
    print("测试 ModelFactory 集成")
    print("="*70)

    # 测试参数
    batch_size = 8
    seq_len = 20
    input_size = 16

    x = torch.randn(batch_size, seq_len, input_size)

    # 测试通过 ModelFactory 创建模型
    model_types = ['lstm', 'gru', 'bilstm', 'bigru']

    for model_type in model_types:
        print(f"\n测试 {model_type.upper()} (通过 ModelFactory)")

        # 加载配置
        config = ConfigLoader.load_model_config(model_type)

        # 创建模型
        model = ModelFactory.create_model(
            model_type=model_type,
            input_size=input_size,
            config=config
        )

        # 前向传播
        model.eval()
        with torch.no_grad():
            output = model(x)

        print(f"  输出形状: {output.shape}")
        assert output.shape == (batch_size, seq_len, 1), \
            f"{model_type} 输出形状错误！"

        print(f"  [OK] {model_type.upper()} 集成成功")

    print("\n" + "="*70)
    print("ModelFactory 集成测试通过！")
    print("="*70)


def test_loss_computation():
    """测试损失函数计算"""
    print("\n" + "="*70)
    print("测试损失函数计算")
    print("="*70)

    batch_size = 16
    seq_len = 40
    input_size = 16

    # 创建测试数据
    x = torch.randn(batch_size, seq_len, input_size)
    targets = torch.rand(batch_size, seq_len, 1)  # Seq2Seq目标

    # 创建模型
    model = LSTM(input_size=input_size, hidden_size=64, num_layers=2)

    # 前向传播
    predictions = model(x)

    # 计算损失
    criterion = torch.nn.MSELoss()
    loss = criterion(predictions, targets)

    print(f"预测形状: {predictions.shape}")
    print(f"目标形状: {targets.shape}")
    print(f"损失值: {loss.item():.6f}")

    assert predictions.shape == targets.shape, "预测和目标形状不匹配！"
    assert not torch.isnan(loss), "损失为NaN！"

    print("  [OK] 损失函数计算正常")

    print("\n" + "="*70)
    print("损失函数测试通过！")
    print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("开始测试 Seq2Seq 模型集成")
    print("="*70)

    # 运行所有测试
    test_baseline_seq2seq()
    test_model_factory_integration()
    test_loss_computation()

    print("\n" + "="*70)
    print("所有测试通过！ Seq2Seq 集成成功！")
    print("="*70)
