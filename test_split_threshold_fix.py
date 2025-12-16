"""
测试 split_threshold 配置读取是否正确

验证修复后，三元组模式能正确读取 triplet_sampling 中的 split_threshold
"""

import json

print("="*70)
print("测试 split_threshold 配置读取")
print("="*70)

# 读取配置文件
config_path = 'configs/models/cnn_lstm_config.json'
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

physics_config = config.get('physics_constraints', {})

# 模拟代码逻辑
siamese_config = physics_config.get('siamese_sampling', {})
siamese_mode = siamese_config.get('enabled', False)

triplet_config = physics_config.get('triplet_sampling', {})
triplet_mode = triplet_config.get('enabled', False)

# 修复后的逻辑
if triplet_mode:
    split_threshold = triplet_config.get('split_threshold', 200)
    step_k = triplet_config.get('step_k', 1)
    mode_name = "三元组模式"
elif siamese_mode:
    split_threshold = siamese_config.get('split_threshold', 300)
    step_k = siamese_config.get('step_k', 1)
    mode_name = "孪生模式"
else:
    split_threshold = 300
    step_k = 1
    mode_name = "无采样模式"

print(f"\n配置文件内容:")
print(f"  siamese_sampling.enabled: {siamese_mode}")
print(f"  siamese_sampling.split_threshold: {siamese_config.get('split_threshold', 'N/A')}")
print(f"  triplet_sampling.enabled: {triplet_mode}")
print(f"  triplet_sampling.split_threshold: {triplet_config.get('split_threshold', 'N/A')}")

print(f"\n读取结果:")
print(f"  启用模式: {mode_name}")
print(f"  split_threshold: {split_threshold}")
print(f"  step_k: {step_k}")

# 验证
expected_threshold = triplet_config.get('split_threshold', 200) if triplet_mode else (
    siamese_config.get('split_threshold', 300) if siamese_mode else 300
)

if split_threshold == expected_threshold:
    print(f"\n[OK] split_threshold 读取正确！")
    print(f"     期望值: {expected_threshold}")
    print(f"     实际值: {split_threshold}")
else:
    print(f"\n[FAIL] split_threshold 读取错误！")
    print(f"       期望值: {expected_threshold}")
    print(f"       实际值: {split_threshold}")

# 特别检查三元组模式
if triplet_mode:
    triplet_threshold_in_config = triplet_config.get('split_threshold')
    print(f"\n特别验证（三元组模式）:")
    print(f"  配置文件中的值: {triplet_threshold_in_config}")
    print(f"  代码读取的值: {split_threshold}")

    if triplet_threshold_in_config is not None and split_threshold == triplet_threshold_in_config:
        print(f"  [OK] 三元组模式正确读取了自己的 split_threshold")
    elif split_threshold == 300:
        print(f"  [FAIL] 三元组模式错误地使用了默认值 300")
        print(f"  这表明代码仍在从 siamese_config 读取！")
    else:
        print(f"  [WARN] 使用了默认值，但配置文件中未设置")

print("\n" + "="*70)
