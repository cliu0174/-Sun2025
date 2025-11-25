"""
验证 Seq2Seq 窗口化修改的正确性
"""
import numpy as np

def apply_windowing(features, targets, window_size, seq2seq=False):
    """复制修改后的窗口化函数"""
    if window_size <= 1:
        return features, targets

    windowed_features = []
    windowed_targets = []

    for i in range(len(features) - window_size):
        window_feat = features[i:i+window_size]
        windowed_features.append(window_feat)

        if seq2seq:
            # Seq2Seq模式：输入X[i:i+window_size]，预测Y[i+1:i+window_size+1]
            window_targ = targets[i+1:i+window_size+1]
            windowed_targets.append(window_targ)
        else:
            # Many-to-One模式：只返回最后一个时间步的目标
            windowed_targets.append(targets[i+window_size-1])

    return np.array(windowed_features), np.array(windowed_targets)


# 创建测试数据
n_samples = 25
window_size = 20
features = np.arange(n_samples).reshape(-1, 1)  # [0, 1, 2, ..., 24]
targets = np.arange(n_samples) * 0.01 + 1.0      # [1.0, 1.01, 1.02, ..., 1.24]

print("="*70)
print("测试数据")
print("="*70)
print(f"原始样本数: {n_samples}")
print(f"窗口大小: {window_size}")
print(f"特征 (cycle编号): {features.squeeze()}")
print(f"目标 (SOH): {targets}")

# 测试 Many-to-One
print("\n" + "="*70)
print("Many-to-One 模式")
print("="*70)
feat_m2o, targ_m2o = apply_windowing(features, targets, window_size, seq2seq=False)
print(f"生成样本数: {len(feat_m2o)}")
print(f"\n前3个样本:")
for i in range(min(3, len(feat_m2o))):
    input_cycles = feat_m2o[i].squeeze()
    output_soh = targ_m2o[i]
    print(f"  样本{i}: 输入 cycle[{int(input_cycles[0])}-{int(input_cycles[-1])}] "
          f"→ 输出 SOH={output_soh:.3f} (cycle {int(input_cycles[-1])})")

# 测试 Many-to-Many
print("\n" + "="*70)
print("Many-to-Many 模式 (Seq2Seq)")
print("="*70)
feat_m2m, targ_m2m = apply_windowing(features, targets, window_size, seq2seq=True)
print(f"生成样本数: {len(feat_m2m)}")
print(f"\n前3个样本:")
for i in range(min(3, len(feat_m2m))):
    input_cycles = feat_m2m[i].squeeze()
    output_sohs = targ_m2m[i]
    print(f"  样本{i}: 输入 cycle[{int(input_cycles[0])}-{int(input_cycles[-1])}]")
    print(f"         → 输出 SOH[{output_sohs[0]:.3f}, ..., {output_sohs[-1]:.3f}] "
          f"(cycle {int(input_cycles[0])+1}-{int(input_cycles[-1])+1})")

# 验证正确性
print("\n" + "="*70)
print("验证结果")
print("="*70)

# 验证1: Many-to-One - 输入最后一个cycle应该等于输出的cycle
success_m2o = True
for i in range(len(feat_m2o)):
    input_last_cycle = int(feat_m2o[i][-1, 0])
    expected_soh = targets[input_last_cycle]
    actual_soh = targ_m2o[i]
    if not np.isclose(expected_soh, actual_soh):
        print(f"❌ Many-to-One 样本{i} 错误: 期望{expected_soh}, 实际{actual_soh}")
        success_m2o = False
        break

if success_m2o:
    print("✅ Many-to-One 验证通过: 输入cycle[0-19] → 输出cycle[19]的SOH")

# 验证2: Many-to-Many - 输出应该是输入的下一个时间步
success_m2m = True
for i in range(len(feat_m2m)):
    input_cycles = feat_m2m[i].squeeze().astype(int)
    expected_output_cycles = input_cycles + 1  # 应该预测下一个时间步
    actual_sohs = targ_m2m[i]
    expected_sohs = targets[expected_output_cycles]

    if not np.allclose(expected_sohs, actual_sohs):
        print(f"❌ Many-to-Many 样本{i} 错误:")
        print(f"   输入cycles: {input_cycles}")
        print(f"   期望输出cycles: {expected_output_cycles}")
        print(f"   期望SOH: {expected_sohs}")
        print(f"   实际SOH: {actual_sohs}")
        success_m2m = False
        break

if success_m2m:
    print("✅ Many-to-Many 验证通过: 输入cycle[0-19] → 输出cycle[1-20]的SOH")

# 验证3: 样本数量
print(f"\n样本数量验证:")
print(f"  Many-to-One: {len(feat_m2o)} (应为 {n_samples - window_size + 1} = 6)")
print(f"  Many-to-Many: {len(feat_m2m)} (应为 {n_samples - window_size} = 5)")

if len(feat_m2o) == n_samples - window_size + 1:
    print("✅ Many-to-One 样本数量正确")
else:
    print("❌ Many-to-One 样本数量错误")

if len(feat_m2m) == n_samples - window_size:
    print("✅ Many-to-Many 样本数量正确")
else:
    print("❌ Many-to-Many 样本数量错误")

print("\n" + "="*70)
print("测试完成!")
print("="*70)
