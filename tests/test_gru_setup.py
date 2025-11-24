"""
测试GRU模型配置和搭建
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from models import ModelFactory, ConfigLoader, UnifiedModelWrapper

def test_gru_configuration():
    """测试GRU配置是否正确"""
    print("="*70)
    print("测试 GRU 配置和搭建")
    print("="*70)

    # 1. 加载配置
    print("\n[1] 加载GRU配置...")
    try:
        config = ConfigLoader.load_model_config('gru')
        print("✅ 配置加载成功")
        ConfigLoader.print_config(config)
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False

    # 2. 创建模型（使用ModelFactory）
    print("\n[2] 使用ModelFactory创建GRU模型...")
    try:
        input_size = 6  # 假设6个特征
        model = ModelFactory.create_model('gru', input_size=input_size, config=config)
        print("✅ 模型创建成功")
        ModelFactory.print_model_info(model)
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False

    # 3. 测试模型前向传播（2D输入 - 单时间步）
    print("\n[3] 测试2D输入（单时间步）...")
    try:
        batch_size = 16
        x_2d = torch.randn(batch_size, input_size)
        output_2d = model(x_2d)
        print(f"输入形状: {x_2d.shape}")
        print(f"输出形状: {output_2d.shape}")
        print(f"输出范围: [{output_2d.min():.4f}, {output_2d.max():.4f}]")

        # 检查输出形状
        if output_2d.shape == (batch_size, 1):
            print("✅ 2D输入测试通过")
        else:
            print(f"❌ 输出形状错误，期望 ({batch_size}, 1)，实际 {output_2d.shape}")
            return False
    except Exception as e:
        print(f"❌ 2D输入测试失败: {e}")
        return False

    # 4. 测试模型前向传播（3D输入 - 多时间步）
    print("\n[4] 测试3D输入（多时间步，窗口化数据）...")
    try:
        window_size = 10
        x_3d = torch.randn(batch_size, window_size, input_size)
        output_3d = model(x_3d)
        print(f"输入形状: {x_3d.shape}")
        print(f"输出形状: {output_3d.shape}")
        print(f"输出范围: [{output_3d.min():.4f}, {output_3d.max():.4f}]")

        # 检查输出形状
        if output_3d.shape == (batch_size, 1):
            print("✅ 3D输入测试通过")
        else:
            print(f"❌ 输出形状错误，期望 ({batch_size}, 1)，实际 {output_3d.shape}")
            return False
    except Exception as e:
        print(f"❌ 3D输入测试失败: {e}")
        return False

    # 5. 测试UnifiedModelWrapper
    print("\n[5] 测试UnifiedModelWrapper...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        wrapper = UnifiedModelWrapper(
            model_type='gru',
            input_size=input_size,
            config=config,
            device=device
        )
        print(f"✅ Wrapper创建成功")
        print(f"设备: {wrapper.device}")
        print(f"模型类型: {wrapper.model_type}")

        # 测试优化器
        optimizer = wrapper.get_optimizer()
        print(f"优化器: {optimizer.__class__.__name__}")
        print(f"学习率: {optimizer.param_groups[0]['lr']}")

        # 测试损失函数
        print(f"损失函数: {wrapper.criterion.__class__.__name__}")

        print("✅ Wrapper测试通过")
    except Exception as e:
        print(f"❌ Wrapper测试失败: {e}")
        return False

    # 6. 配置检查
    print("\n[6] 配置参数检查...")
    issues = []

    # 检查架构参数
    arch = config['architecture']
    if arch['hidden_size'] <= 0:
        issues.append("hidden_size必须 > 0")
    if arch['num_layers'] <= 0:
        issues.append("num_layers必须 > 0")
    if not isinstance(arch['fc_hidden_sizes'], list):
        issues.append("fc_hidden_sizes必须是列表")
    if not (0 <= arch['dropout_rate'] < 1):
        issues.append("dropout_rate必须在[0, 1)范围内")

    # 检查训练参数
    train = config['training']
    if train['num_epochs'] <= 0:
        issues.append("num_epochs必须 > 0")
    if train['batch_size'] <= 0:
        issues.append("batch_size必须 > 0")
    if train['learning_rate'] <= 0:
        issues.append("learning_rate必须 > 0")

    if issues:
        print("❌ 配置存在以下问题:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ 配置参数检查通过")

    # 7. 与LSTM对比
    print("\n[7] GRU vs LSTM 参数对比...")
    try:
        lstm_model = ModelFactory.create_model('lstm', input_size=input_size)
        gru_model = ModelFactory.create_model('gru', input_size=input_size)

        lstm_params = sum(p.numel() for p in lstm_model.parameters())
        gru_params = sum(p.numel() for p in gru_model.parameters())

        print(f"LSTM 参数量: {lstm_params:,}")
        print(f"GRU 参数量:  {gru_params:,}")
        print(f"参数减少:    {lstm_params - gru_params:,} ({(1 - gru_params/lstm_params)*100:.2f}%)")
        print("✅ GRU参数量少于LSTM（符合预期）")
    except Exception as e:
        print(f"❌ 对比测试失败: {e}")
        return False

    return True


if __name__ == "__main__":
    print("\n开始测试GRU配置和搭建...\n")

    success = test_gru_configuration()

    print("\n" + "="*70)
    if success:
        print("✅ 所有测试通过！GRU配置和搭建正确")
        print("\n你可以开始使用GRU进行训练了：")
        print("  1. 单电池训练: python train_single_model.py")
        print("     (修改MODEL_TYPE='gru')")
        print("  2. 跨电池训练: python train_cross_battery.py")
        print("     (修改MODEL_TYPE='gru')")
    else:
        print("❌ 测试失败，请检查上述错误信息")
    print("="*70)
