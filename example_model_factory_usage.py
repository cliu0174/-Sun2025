"""
示例脚本：展示如何使用统一的模型工厂系统。

这个脚本展示了三种使用模型工厂的方式：
1. 基础用法：使用默认配置
2. 进阶用法：覆盖配置参数
3. 高级用法：使用UnifiedModelWrapper进行完整的训练流程
"""

import torch
from models import ModelFactory, ConfigLoader, UnifiedModelWrapper


def example_1_basic_usage():
    """示例1: 基础用法 - 使用默认配置创建模型"""
    print("\n" + "=" * 70)
    print("示例1: 基础用法 - 使用默认配置")
    print("=" * 70)

    # 创建不同类型的模型
    model_types = ['fnn', 'cnn', 'lstm', 'bpinn']
    input_size = 6  # 假设有6个输入特征
    batch_size = 16

    for model_type in model_types:
        print(f"\n--- 创建 {model_type.upper()} 模型 ---")

        # 方式1: 直接使用ModelFactory创建模型（自动加载配置）
        model = ModelFactory.create_model(
            model_type=model_type,
            input_size=input_size
        )

        # 测试前向传播
        test_input = torch.randn(batch_size, input_size)
        output = model(test_input)

        # 打印模型信息
        ModelFactory.print_model_info(model)
        print(f"输入形状: {test_input.shape}")
        print(f"输出形状: {output.shape}")


def example_2_custom_parameters():
    """示例2: 进阶用法 - 覆盖配置参数"""
    print("\n" + "=" * 70)
    print("示例2: 进阶用法 - 覆盖配置参数")
    print("=" * 70)

    input_size = 6

    # 方式2a: 先加载配置，再修改
    print("\n--- 方法A: 先加载配置，再修改 ---")
    config = ConfigLoader.load_model_config('fnn')
    ConfigLoader.print_config(config)

    # 修改配置
    config['architecture']['hidden_sizes'] = [128, 64, 32]
    config['architecture']['dropout_rate'] = 0.3

    model = ModelFactory.create_model(
        model_type='fnn',
        input_size=input_size,
        config=config
    )
    ModelFactory.print_model_info(model)

    # 方式2b: 直接传入覆盖参数
    print("\n--- 方法B: 直接传入覆盖参数 ---")
    model = ModelFactory.create_model(
        model_type='fnn',
        input_size=input_size,
        hidden_sizes=[256, 128, 64, 32],  # 覆盖配置中的hidden_sizes
        dropout_rate=0.1  # 覆盖配置中的dropout_rate
    )
    ModelFactory.print_model_info(model)


def example_3_unified_wrapper():
    """示例3: 高级用法 - 使用UnifiedModelWrapper"""
    print("\n" + "=" * 70)
    print("示例3: 高级用法 - UnifiedModelWrapper")
    print("=" * 70)

    input_size = 6

    # 创建不同模型的wrapper
    print("\n--- 创建FNN模型wrapper ---")
    fnn_wrapper = UnifiedModelWrapper(
        model_type='fnn',
        input_size=input_size,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"模型: {fnn_wrapper}")
    ModelFactory.print_model_info(fnn_wrapper.model)

    # 获取优化器
    optimizer = fnn_wrapper.get_optimizer()
    print(f"\n优化器: {optimizer.__class__.__name__}")
    print(f"学习率: {optimizer.param_groups[0]['lr']}")

    # 获取损失函数
    print(f"损失函数: {fnn_wrapper.criterion.__class__.__name__}")

    # 创建BPINN模型的wrapper
    print("\n--- 创建BPINN模型wrapper ---")
    bpinn_wrapper = UnifiedModelWrapper(
        model_type='bpinn',
        input_size=input_size,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"模型: {bpinn_wrapper}")
    ModelFactory.print_model_info(bpinn_wrapper.model)
    print(f"损失函数: {bpinn_wrapper.criterion.__class__.__name__}")


def example_4_checkpoint_saving():
    """示例4: 检查点保存和加载"""
    print("\n" + "=" * 70)
    print("示例4: 检查点保存和加载")
    print("=" * 70)

    input_size = 6

    # 创建模型
    print("\n--- 创建并保存模型 ---")
    wrapper = UnifiedModelWrapper(
        model_type='fnn',
        input_size=input_size
    )

    # 模拟训练状态
    wrapper.best_mae = 0.025
    wrapper.best_epoch = 100
    wrapper.training_history = {
        'train_loss': [0.1, 0.08, 0.06],
        'test_mae': [0.04, 0.03, 0.025],
        'test_rmse': [0.05, 0.04, 0.03]
    }

    # 保存检查点
    checkpoint_path = 'test_checkpoint.pth'
    optimizer = wrapper.get_optimizer()
    wrapper.save_checkpoint(
        save_path=checkpoint_path,
        epoch=100,
        optimizer_state=optimizer.state_dict()
    )

    # 加载检查点
    print("\n--- 加载检查点 ---")
    loaded_wrapper = UnifiedModelWrapper.load_checkpoint(checkpoint_path)
    print(f"加载的模型: {loaded_wrapper}")
    print(f"最佳MAE: {loaded_wrapper.best_mae:.6f}")
    print(f"最佳Epoch: {loaded_wrapper.best_epoch}")
    print(f"训练历史: {list(loaded_wrapper.training_history.keys())}")


def example_5_compare_all_models():
    """示例5: 对比所有模型的架构"""
    print("\n" + "=" * 70)
    print("示例5: 对比所有模型")
    print("=" * 70)

    input_size = 6
    models_info = []

    for model_type in ModelFactory.SUPPORTED_MODELS:
        model = ModelFactory.create_model(model_type, input_size)
        info = ModelFactory.get_model_info(model)
        info['model_type'] = model_type.upper()
        models_info.append(info)

    # 打印对比表格
    print("\n模型对比:")
    print(f"{'模型类型':<15} {'总参数':<15} {'模型大小(MB)':<15}")
    print("-" * 50)
    for info in models_info:
        print(f"{info['model_type']:<15} "
              f"{info['total_parameters']:<15,} "
              f"{info['model_size_mb']:<15.2f}")


def example_6_create_loss_functions():
    """示例6: 创建不同的损失函数"""
    print("\n" + "=" * 70)
    print("示例6: 创建损失函数")
    print("=" * 70)

    # 标准模型使用MSELoss
    for model_type in ['fnn', 'cnn', 'lstm']:
        criterion = ModelFactory.create_loss_function(model_type)
        print(f"{model_type.upper()}: {criterion.__class__.__name__}")

    # BPINN使用特殊的损失函数
    bpinn_criterion = ModelFactory.create_loss_function(
        'bpinn',
        lambda_physics=0.02  # 可以覆盖默认参数
    )
    print(f"BPINN: {bpinn_criterion.__class__.__name__} "
          f"(lambda_physics={bpinn_criterion.lambda_physics})")


if __name__ == "__main__":
    """运行所有示例"""

    print("\n" + "=" * 70)
    print("统一模型工厂系统 - 使用示例")
    print("=" * 70)

    try:
        # 运行所有示例
        example_1_basic_usage()
        example_2_custom_parameters()
        example_3_unified_wrapper()
        example_4_checkpoint_saving()
        example_5_compare_all_models()
        example_6_create_loss_functions()

        print("\n" + "=" * 70)
        print("所有示例运行完成!")
        print("=" * 70)

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
