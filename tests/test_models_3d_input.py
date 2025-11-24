"""
测试模型对 3D 窗口化输入的兼容性。
"""

import torch
from models.baseline_models import FNN, CNN, LSTM, GRU, MLP, ResCNN


def test_models_3d_input():
    """测试所有模型对 3D 输入的处理"""
    
    batch_size = 16
    input_size = 6
    window_size = 10
    
    # 2D 输入: (batch_size, input_size)
    x_2d = torch.randn(batch_size, input_size)
    
    # 3D 输入: (batch_size, window_size, input_size)
    x_3d = torch.randn(batch_size, window_size, input_size)
    
    print("=" * 70)
    print("测试模型对 2D 和 3D 输入的兼容性")
    print("=" * 70)
    print(f"2D 输入形状: {x_2d.shape} (batch, features)")
    print(f"3D 输入形状: {x_3d.shape} (batch, window_size, features)")
    print()
    
    models = {
        'FNN': FNN(input_size=input_size),
        'CNN': CNN(input_size=input_size),
        'LSTM': LSTM(input_size=input_size),
        'GRU': GRU(input_size=input_size),
        'MLP': MLP(input_size=input_size),
        'ResCNN': ResCNN(input_size=input_size),
    }
    
    for name, model in models.items():
        try:
            # 测试 2D 输入
            with torch.no_grad():
                out_2d = model(x_2d)
            
            # 测试 3D 输入（窗口化）
            with torch.no_grad():
                out_3d = model(x_3d)
            
            print(f"✅ {name:10s} - 2D输入: {x_2d.shape} → {out_2d.shape}")
            print(f"           3D输入: {x_3d.shape} → {out_3d.shape}")
            
            # 验证输出形状都是 (batch_size, 1)
            assert out_2d.shape == (batch_size, 1), f"{name} 2D输出形状不正确"
            assert out_3d.shape == (batch_size, 1), f"{name} 3D输出形状不正确"
            
        except Exception as e:
            print(f"❌ {name:10s} - 错误: {str(e)}")
        
        print()
    
    print("=" * 70)
    print("✅ 所有模型兼容性测试通过！")
    print("=" * 70)

if __name__ == "__main__":
    test_models_3d_input()
