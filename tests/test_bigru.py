"""快速测试BiGRU模型"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from models import ModelFactory, ConfigLoader

print("="*70)
print("测试 BiGRU 模型")
print("="*70)

# 1. 加载配置
config = ConfigLoader.load_model_config('bigru')
print(f"\n✓ 配置加载成功")
print(f"  Window Size: {config['data']['window_size']}")
print(f"  Hidden Size: {config['architecture']['hidden_size']}")
print(f"  Num Layers: {config['architecture']['num_layers']}")
print(f"  FC Layers: {config['architecture']['fc_hidden_sizes']}")

# 2. 创建模型
input_size = 6
model = ModelFactory.create_model('bigru', input_size=input_size, config=config)
print(f"\n✓ 模型创建成功")

# 3. 测试前向传播
batch_size = 16
window_size = 40
x = torch.randn(batch_size, window_size, input_size)
output = model(x)

print(f"\n✓ 前向传播测试")
print(f"  输入形状: {x.shape}")
print(f"  输出形状: {output.shape}")
print(f"  输出范围: [{output.min():.4f}, {output.max():.4f}]")

# 4. 参数量对比
from models import GRU
gru_model = GRU(input_size=input_size, hidden_size=64, num_layers=2, fc_hidden_sizes=[64])
bigru_params = sum(p.numel() for p in model.parameters())
gru_params = sum(p.numel() for p in gru_model.parameters())

print(f"\n✓ 参数量对比")
print(f"  GRU:   {gru_params:,}")
print(f"  BiGRU: {bigru_params:,}")
print(f"  增加:  {bigru_params - gru_params:,} ({(bigru_params/gru_params - 1)*100:.1f}%)")

print("\n" + "="*70)
print("✅ BiGRU 测试通过！可以开始训练")
print("\n运行命令: python train_cross_battery.py")
print("="*70)
