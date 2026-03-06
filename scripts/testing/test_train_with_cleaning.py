"""
测试训练脚本是否支持数据清洗参数
"""

from train_cross_battery import train_cross_battery_model

print("="*70)
print("测试：训练脚本 + 数据清洗集成")
print("="*70)

# 测试1: 不清洗（默认行为）
print("\n[测试1] 默认行为（不清洗）")
print("-"*70)
try:
    wrapper, results, data_dict = train_cross_battery_model(
        model_type='gru',
        device='cuda',
        apply_cleaning=False  # 显式指定不清洗
    )
    print(f"\n[OK] 不清洗模式训练成功")
    print(f"  Test MAE: {results['test_mae']*100:.4f}%")
except KeyboardInterrupt:
    print("\n[中断] 用户停止训练")
except Exception as e:
    print(f"\n[ERROR] 训练失败: {e}")

# 测试2: 启用清洗
print("\n" + "="*70)
print("[测试2] 启用数据清洗")
print("-"*70)

user_input = input("\n是否要测试清洗模式？这将再次训练模型（y/n）: ")

if user_input.lower() == 'y':
    try:
        wrapper, results, data_dict = train_cross_battery_model(
            model_type='gru',
            device='cuda',
            apply_cleaning=True  # 启用清洗
        )
        print(f"\n[OK] 清洗模式训练成功")
        print(f"  Test MAE: {results['test_mae']*100:.4f}%")
    except KeyboardInterrupt:
        print("\n[中断] 用户停止训练")
    except Exception as e:
        print(f"\n[ERROR] 训练失败: {e}")
else:
    print("\n[跳过] 清洗模式测试")

print("\n" + "="*70)
print("测试完成")
print("="*70)
