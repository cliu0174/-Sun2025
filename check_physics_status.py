"""
检查物理约束是否在训练中被启用
"""

import sys
import argparse
from models import ConfigLoader

def check_physics_status(model_type='lstm'):
    print("="*70)
    print("物理约束状态检查")
    print("="*70)

    # 加载配置
    config = ConfigLoader.load_model_config(model_type)

    print(f"\n模型类型: {model_type}")
    print(f"\n完整配置:")

    # 检查物理约束配置
    physics_config = config.get('physics_constraints', {})

    if not physics_config:
        print("\n❌ 错误：配置文件中没有 'physics_constraints' 字段！")
        return False

    print(f"\n物理约束配置:")
    for key, value in physics_config.items():
        if key == 'temporal_decay':
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

    # 检查关键参数
    enabled = physics_config.get('enabled', False)

    print(f"\n关键检查:")
    print(f"  ✓ 'enabled' = {enabled}")

    if not enabled:
        print("\n❌ 物理约束未启用！")
        print("请在配置文件中设置 'enabled': true")
        return False

    # 检查权重
    base_weight = physics_config.get('base_loss_weight', 1.0)
    mono_weight = physics_config.get('monotonic_weight', 0.1)

    print(f"  ✓ base_loss_weight = {base_weight}")
    print(f"  ✓ monotonic_weight = {mono_weight}")

    if mono_weight < 0.001:
        print("\n⚠️  警告：monotonic_weight 太小，可能没有效果")

    ratio = mono_weight / base_weight
    print(f"\n权重比例: monotonic_weight / base_loss_weight = {ratio:.3f}")

    if ratio < 0.1:
        print("⚠️  警告：物理约束权重相对基础损失太小")
        print(f"   建议将 monotonic_weight 增加到至少 {base_weight * 0.1:.1f}")

    print("\n" + "="*70)
    print("下一步：检查训练过程")
    print("="*70)

    print("\n请在训练开始时查看日志，确认以下内容：")
    print("  1. 是否显示 '使用物理约束模式'")
    print("  2. 是否显示 '损失函数: PhysicsConstrainedLoss (物理约束)'")
    print("  3. DataLoader 的 collate_fn 是否为 custom_collate_fn")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lstm',
                       help='模型类型 (lstm/gru/bilstm/bigru)')
    args = parser.parse_args()

    check_physics_status(args.model)
