"""
测试 Many-to-One 和 Many-to-Many 模型共存

验证：
1. Many-to-One 模型输出 (batch, 1)
2. Many-to-Many 模型输出 (batch, seq_len, 1)
3. ModelFactory 正确创建两种模型
4. 配置文件正确加载
"""

import torch
from models import (
    LSTM, GRU, BiLSTM, BiGRU,
    LSTMManyToMany, GRUManyToMany, BiLSTMManyToMany, BiGRUManyToMany,
    ModelFactory, ConfigLoader
)

def test_many_to_one_models():
    """测试 Many-to-One 模型"""
    print("="*70)
    print("测试 Many-to-One 模型")
    print("="*70)

    batch_size = 16
    seq_len = 40
    input_size = 16

    # 创建测试数据 (batch, seq_len, input_size)
    x = torch.randn(batch_size, seq_len, input_size)
    print(f"\n输入形状: {x.shape}")

    models = {
        'LSTM': LSTM(input_size=input_size, hidden_size=64, num_layers=2),
        'GRU': GRU(input_size=input_size, hidden_size=64, num_layers=2),
        'BiLSTM': BiLSTM(input_size=input_size, hidden_size=64, num_layers=2),
        'BiGRU': BiGRU(input_size=input_size, hidden_size=64, num_layers=2),
    }

    for name, model in models.items():
        print(f"\n{'-'*70}")
        print(f"测试 {name} (Many-to-One)")
        print(f"{'-'*70}")

        model.eval()
        with torch.no_grad():
            output = model(x)

        print(f"输出形状: {output.shape}")
        print(f"  预期: ({batch_size}, 1)")
        print(f"  实际: {tuple(output.shape)}")

        # 验证输出形状
        assert output.shape == (batch_size, 1), \
            f"{name} 输出形状错误！预期 ({batch_size}, 1)，实际 {output.shape}"

        # 验证输出范围
        print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        assert output.min() >= 0 and output.max() <= 1, \
            f"{name} 输出范围错误！应在[0,1]之间"

        print(f"  [OK] {name} Many-to-One 测试通过")

    print("\n" + "="*70)
    print("所有 Many-to-One 模型测试通过！")
    print("="*70)


def test_many_to_many_models():
    """测试 Many-to-Many 模型"""
    print("\n" + "="*70)
    print("测试 Many-to-Many 模型")
    print("="*70)

    batch_size = 16
    seq_len = 40
    input_size = 16

    # 创建测试数据 (batch, seq_len, input_size)
    x = torch.randn(batch_size, seq_len, input_size)
    print(f"\n输入形状: {x.shape}")

    models = {
        'LSTMManyToMany': LSTMManyToMany(input_size=input_size, hidden_size=64, num_layers=2),
        'GRUManyToMany': GRUManyToMany(input_size=input_size, hidden_size=64, num_layers=2),
        'BiLSTMManyToMany': BiLSTMManyToMany(input_size=input_size, hidden_size=64, num_layers=2),
        'BiGRUManyToMany': BiGRUManyToMany(input_size=input_size, hidden_size=64, num_layers=2),
    }

    for name, model in models.items():
        print(f"\n{'-'*70}")
        print(f"测试 {name} (Many-to-Many)")
        print(f"{'-'*70}")

        model.eval()
        with torch.no_grad():
            output = model(x)

        print(f"输出形状: {output.shape}")
        print(f"  预期: ({batch_size}, {seq_len}, 1)")
        print(f"  实际: {tuple(output.shape)}")

        # 验证输出形状
        assert output.shape == (batch_size, seq_len, 1), \
            f"{name} 输出形状错误！预期 ({batch_size}, {seq_len}, 1)，实际 {output.shape}"

        # 验证输出范围
        print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        assert output.min() >= 0 and output.max() <= 1, \
            f"{name} 输出范围错误！应在[0,1]之间"

        print(f"  [OK] {name} Many-to-Many 测试通过")

    print("\n" + "="*70)
    print("所有 Many-to-Many 模型测试通过！")
    print("="*70)


def test_model_factory():
    """测试 ModelFactory 创建两种模型"""
    print("\n" + "="*70)
    print("测试 ModelFactory")
    print("="*70)

    batch_size = 8
    seq_len = 20
    input_size = 16

    x = torch.randn(batch_size, seq_len, input_size)

    # 测试 Many-to-One 模型
    print("\n测试 Many-to-One 模型创建:")
    for model_type in ['lstm', 'gru', 'bilstm', 'bigru']:
        print(f"  创建 {model_type.upper()}...", end=" ")
        config = ConfigLoader.load_model_config(model_type)
        model = ModelFactory.create_model(model_type, input_size=input_size, config=config)
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert output.shape == (batch_size, 1), f"{model_type} Many-to-One 输出形状错误！"
        print(f"[OK] 输出形状: {output.shape}")

    # 测试 Many-to-Many 模型
    print("\n测试 Many-to-Many 模型创建:")
    for model_type in ['lstm_manytomany', 'gru_manytomany', 'bilstm_manytomany', 'bigru_manytomany']:
        print(f"  创建 {model_type.upper()}...", end=" ")
        config = ConfigLoader.load_model_config(model_type)
        model = ModelFactory.create_model(model_type, input_size=input_size, config=config)
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert output.shape == (batch_size, seq_len, 1), f"{model_type} Many-to-Many 输出形状错误！"
        print(f"[OK] 输出形状: {output.shape}")

    print("\n" + "="*70)
    print("ModelFactory 测试通过！")
    print("="*70)


def test_loss_computation():
    """测试两种模型的损失计算"""
    print("\n" + "="*70)
    print("测试损失函数计算")
    print("="*70)

    batch_size = 16
    seq_len = 40
    input_size = 16

    x = torch.randn(batch_size, seq_len, input_size)
    criterion = torch.nn.MSELoss()

    # Many-to-One
    print("\n测试 Many-to-One 损失:")
    model_m2o = LSTM(input_size=input_size, hidden_size=64, num_layers=2)
    targets_m2o = torch.rand(batch_size, 1)  # (batch, 1)

    predictions_m2o = model_m2o(x)
    loss_m2o = criterion(predictions_m2o, targets_m2o)

    print(f"  预测形状: {predictions_m2o.shape}")
    print(f"  目标形状: {targets_m2o.shape}")
    print(f"  损失值: {loss_m2o.item():.6f}")
    assert not torch.isnan(loss_m2o), "Many-to-One 损失为NaN！"
    print("  [OK] Many-to-One 损失计算正常")

    # Many-to-Many
    print("\n测试 Many-to-Many 损失:")
    model_m2m = LSTMManyToMany(input_size=input_size, hidden_size=64, num_layers=2)
    targets_m2m = torch.rand(batch_size, seq_len, 1)  # (batch, seq_len, 1)

    predictions_m2m = model_m2m(x)
    loss_m2m = criterion(predictions_m2m, targets_m2m)

    print(f"  预测形状: {predictions_m2m.shape}")
    print(f"  目标形状: {targets_m2m.shape}")
    print(f"  损失值: {loss_m2m.item():.6f}")
    assert not torch.isnan(loss_m2m), "Many-to-Many 损失为NaN！"
    print("  [OK] Many-to-Many 损失计算正常")

    print("\n" + "="*70)
    print("损失函数测试通过！")
    print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("开始测试 Many-to-One 和 Many-to-Many 模型共存")
    print("="*70)

    # 运行所有测试
    test_many_to_one_models()
    test_many_to_many_models()
    test_model_factory()
    test_loss_computation()

    print("\n" + "="*70)
    print("所有测试通过！Many-to-One 和 Many-to-Many 模型共存成功！")
    print("="*70)
    print("\n总结:")
    print("  - Many-to-One 模型: LSTM, GRU, BiLSTM, BiGRU")
    print("    输出形状: (batch, 1)")
    print("  - Many-to-Many 模型: LSTMManyToMany, GRUManyToMany, BiLSTMManyToMany, BiGRUManyToMany")
    print("    输出形状: (batch, seq_len, 1)")
    print("  - ModelFactory 支持创建两种类型的模型")
    print("  - train_cross_battery.py 已更新以支持两种模型")
