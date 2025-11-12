"""
训练诊断脚本 - 帮助诊断和改进模型训练问题

用于检查：
1. 特征数量是否正确
2. 学习曲线是否平坦
3. 梯度是否正常
4. 建议调整参数
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader_hust import load_single_hust_battery
from src.feature_selector import get_top_correlated_features, create_filtered_data_dict
from src.model_trainer import ConfigLoader, ModelFactory, ModelTrainer
from src.baseline_models import FNN


def diagnose_training(battery_name='1-1', model_name='FNN', epochs=200):
    """
    诊断训练问题的详细脚本。
    
    Args:
        battery_name: 电池名称
        model_name: 模型名称
        epochs: 诊断用的训练轮数
    """
    print("\n" + "=" * 80)
    print("🔍 模型训练诊断工具")
    print("=" * 80)
    
    # 1. 加载配置和数据
    print(f"\n[1/5] 加载配置和数据...")
    config = ConfigLoader.load_model_config_by_name(model_name)
    
    battery_file = os.path.join('data/HUST data', f'{battery_name}.csv')
    if not os.path.exists(battery_file):
        print(f"❌ 文件不存在: {battery_file}")
        return
    
    data_dict = load_single_hust_battery(battery_file, train_ratio=0.75, normalize_target=True)
    print(f"  ✓ 加载完成")
    print(f"    原始特征数: {len(data_dict['feature_names'])}")
    print(f"    训练样本: {data_dict['n_train']}")
    print(f"    测试样本: {data_dict['n_test']}")
    
    # 2. 特征筛选分析
    print(f"\n[2/5] 特征筛选分析...")
    threshold = config['feature_selection']['correlation_threshold']
    selection_result = get_top_correlated_features(data_dict, threshold)
    
    print(f"  ✓ 特征筛选完成")
    print(f"    筛选阈值: {threshold}")
    print(f"    保留特征数: {selection_result['num_features']} / {len(data_dict['feature_names'])}")
    print(f"    特征索引: {selection_result['selected_indices']}")
    
    # 创建筛选后的数据
    filtered_data = create_filtered_data_dict(data_dict, selection_result['selected_indices'])
    input_size = filtered_data['train_features'].shape[1]
    
    print(f"\n⚠️ 配置检查:")
    print(f"    配置文件中 input_size: {config['architecture']['input_size']}")
    print(f"    实际输入特征维度: {input_size}")
    if config['architecture']['input_size'] != input_size:
        print(f"    ⚠️ 警告: 输入维度不匹配! 应该使用 input_size: {input_size}")
    
    # 3. 创建模型
    print(f"\n[3/5] 创建模型...")
    model = ModelFactory.create_model(config, input_size)
    device = torch.device('cpu')
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ 模型创建完成")
    print(f"    模型类型: {model_name}")
    print(f"    总参数数: {total_params}")
    
    # 4. 模拟训练并记录详细信息
    print(f"\n[4/5] 开始诊断训练 (用{epochs}个epoch)...")
    
    from src.data_loader_hust import create_hust_dataloaders
    
    train_loader, test_loader = create_hust_dataloaders(
        filtered_data,
        batch_size=config['training']['batch_size']
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    
    # 记录详细的训练过程
    history = {
        'train_loss': [],
        'test_mae': [],
        'test_rmse': [],
        'grad_norm': [],
        'weight_norm': []
    }
    
    best_mae = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        
        for features, capacity in train_loader:
            features = features.to(device)
            capacity = capacity.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, capacity)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # 计算梯度范数
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = np.sqrt(grad_norm)
        
        # 计算权重范数
        weight_norm = 0.0
        for p in model.parameters():
            weight_norm += p.data.norm(2).item() ** 2
        weight_norm = np.sqrt(weight_norm)
        
        # 评估
        model.eval()
        test_mae = 0.0
        test_rmse = 0.0
        
        with torch.no_grad():
            for features, capacity in test_loader:
                features = features.to(device)
                capacity = capacity.to(device).unsqueeze(1)
                
                predictions = model(features)
                
                mae = torch.mean(torch.abs(predictions - capacity)).item()
                rmse = torch.sqrt(torch.mean((predictions - capacity) ** 2)).item()
                
                test_mae += mae * features.size(0)
                test_rmse += rmse * features.size(0)
        
        mae = test_mae / len(test_loader.dataset)
        rmse = test_rmse / len(test_loader.dataset)
        
        history['train_loss'].append(train_loss)
        history['test_mae'].append(mae)
        history['test_rmse'].append(rmse)
        history['grad_norm'].append(grad_norm)
        history['weight_norm'].append(weight_norm)
        
        # 更新最佳模型
        if mae < best_mae:
            best_mae = mae
            best_epoch = epoch + 1
        
        # 打印信息
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:3d}: Loss={train_loss:.6f}, MAE={mae:.6f}, "
                  f"GradNorm={grad_norm:.6e}, Best={best_epoch}")
    
    # 5. 分析结果
    print(f"\n[5/5] 分析诊断结果...")
    
    print(f"\n📊 训练结果汇总:")
    print(f"  最终MAE: {history['test_mae'][-1]*100:.4f}%")
    print(f"  最终RMSE: {history['test_rmse'][-1]*100:.4f}%")
    print(f"  最佳Epoch: {best_epoch}")
    print(f"  最佳MAE: {best_mae*100:.4f}%")
    print(f"  最终梯度范数: {history['grad_norm'][-1]:.6e}")
    print(f"  最终权重范数: {history['weight_norm'][-1]:.6f}")
    
    # 分析学习曲线
    print(f"\n📈 学习曲线分析:")
    
    # 检查MAE是否在早期停滞
    early_mae = np.mean(history['test_mae'][:10]) if len(history['test_mae']) >= 10 else history['test_mae'][0]
    late_mae = np.mean(history['test_mae'][-10:]) if len(history['test_mae']) >= 10 else history['test_mae'][-1]
    mae_improvement = (early_mae - late_mae) / early_mae * 100 if early_mae > 0 else 0
    
    print(f"  早期平均MAE (前10个epoch): {early_mae*100:.4f}%")
    print(f"  后期平均MAE (后10个epoch): {late_mae*100:.4f}%")
    print(f"  MAE改进百分比: {mae_improvement:.2f}%")
    
    if mae_improvement < 1:
        print(f"  ⚠️ 警告: MAE改进不足1%，模型可能在早期就停滞了")
    elif mae_improvement < 5:
        print(f"  ⚠️ 注意: MAE改进较小，可能需要调整参数")
    else:
        print(f"  ✓ 模型持续改进中")
    
    # 检查梯度
    print(f"\n🔄 梯度分析:")
    early_grad = np.mean(history['grad_norm'][:10]) if len(history['grad_norm']) >= 10 else history['grad_norm'][0]
    late_grad = np.mean(history['grad_norm'][-10:]) if len(history['grad_norm']) >= 10 else history['grad_norm'][-1]
    
    print(f"  早期梯度范数: {early_grad:.6e}")
    print(f"  后期梯度范数: {late_grad:.6e}")
    
    if late_grad < 1e-7:
        print(f"  ⚠️ 警告: 梯度过小，模型无法继续学习（梯度消失）")
    elif late_grad > 10:
        print(f"  ⚠️ 警告: 梯度过大，可能存在梯度爆炸")
    else:
        print(f"  ✓ 梯度范围正常")
    
    # 建议
    print(f"\n💡 调整建议:")
    
    recommendations = []
    
    if best_epoch <= 5:
        recommendations.append("• MAE在早期(≤5 epoch)就最优了，说明：")
        recommendations.append("  - 可能学习率过高，导致跳过最优点")
        recommendations.append("  - 解决方案: 尝试降低学习率 (--lr 0.0005 或 0.0003)")
    
    if mae_improvement < 1:
        recommendations.append("• 模型在早期就停滞，没有继续改进：")
        recommendations.append("  - 可能网络容量不足")
        recommendations.append("  - 解决方案: 增加隐藏层大小，如改为 [128, 64, 32]")
    
    if late_grad < 1e-7:
        recommendations.append("• 梯度接近于0，存在梯度消失：")
        recommendations.append("  - 可能Dropout率过高或网络太深")
        recommendations.append("  - 解决方案: 降低Dropout率 (0.1 或关闭)")
    
    if len(data_dict['train_capacity']) < 1000:
        recommendations.append("• 训练样本较少：")
        recommendations.append("  - 可能容易过拟合")
        recommendations.append("  - 解决方案: 增加Dropout或使用L2正则化")
    
    if selection_result['num_features'] < 8:
        recommendations.append("• 筛选后特征数较少：")
        recommendations.append("  - 可能丢失了重要信息")
        recommendations.append("  - 解决方案: 降低相关性阈值 (--corr-threshold 0.3)")
    
    if recommendations:
        print("\n".join(recommendations))
    else:
        print("✓ 模型训练看起来正常，没有明显问题")
    
    # 绘制学习曲线
    print(f"\n📉 生成学习曲线图表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # MAE曲线
    ax = axes[0, 0]
    ax.plot(history['test_mae'], 'b-', label='Test MAE', linewidth=2)
    ax.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best Epoch: {best_epoch}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.set_title('Test MAE vs Epoch')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Loss曲线
    ax = axes[0, 1]
    ax.plot(history['train_loss'], 'g-', label='Train Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss vs Epoch')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 梯度范数
    ax = axes[1, 0]
    ax.semilogy(history['grad_norm'], 'r-', label='Gradient Norm', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm (log scale)')
    ax.set_title('Gradient Norm vs Epoch')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 权重范数
    ax = axes[1, 1]
    ax.plot(history['weight_norm'], 'purple', label='Weight Norm', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weight Norm')
    ax.set_title('Weight Norm vs Epoch')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plot_path = f'results/diagnosis_{model_name}_{battery_name}.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f"  ✓ 图表已保存: {plot_path}")
    plt.close()
    
    print("\n" + "=" * 80)
    print("✅ 诊断完成")
    print("=" * 80)
    
    return history, best_epoch


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='模型训练诊断工具')
    parser.add_argument('--battery', type=str, default='1-1',
                        help='电池名称 (默认: 1-1)')
    parser.add_argument('--model', type=str, default='FNN',
                        choices=['FNN', 'CNN', 'LSTM'],
                        help='模型名称 (默认: FNN)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='诊断训练的轮数 (默认: 200)')
    
    args = parser.parse_args()
    
    diagnose_training(args.battery, args.model, args.epochs)
