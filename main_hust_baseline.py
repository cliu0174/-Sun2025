"""
HUST数据集基准模型统一训练脚本

通过指定模型名称和配置，灵活训练FNN、CNN或LSTM。

使用方法:
    # 训练FNN (使用默认配置)
    python main_hust_baseline.py --model FNN --battery 1-1
    
    # 训练CNN (自定义学习率)
    python main_hust_baseline.py --model CNN --battery 1-1 --lr 0.0005
    
    # 训练LSTM (自定义批量大小)
    python main_hust_baseline.py --model LSTM --battery 1-1 --batch-size 32
    
    # 训练多个电池
    python main_hust_baseline.py --model FNN --batteries 1-1 1-2 1-3
"""

import os
import sys
import json
import torch
import random
import numpy as np
import argparse
import pickle
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader_hust import load_single_hust_battery
from src.model_trainer import ConfigLoader, ModelTrainer
from src.utils import ensure_dir, print_metrics


def set_seed(seed=42):
    """设置随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def override_config(config, args):
    """
    用命令行参数覆盖配置。
    
    Args:
        config: 原始配置字典
        args: 命令行参数
        
    Returns:
        修改后的配置字典
    """
    # 学习率
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
        print(f"  学习率已修改: {args.lr}")
    
    # 批量大小
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
        print(f"  批量大小已修改: {args.batch_size}")
    
    # 轮数
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
        print(f"  训练轮数已修改: {args.epochs}")
    
    # 相关性阈值
    if args.corr_threshold is not None:
        config['feature_selection']['correlation_threshold'] = args.corr_threshold
        print(f"  相关性阈值已修改: {args.corr_threshold}")
    
    return config


def train_battery(model_name, battery_name, config, device, results_dir):
    """
    训练单个电池。
    
    Args:
        model_name: 模型名称 ('FNN', 'CNN', 'LSTM')
        battery_name: 电池名称 (例如'1-1')
        config: 配置字典
        device: 计算设备
        results_dir: 结果保存目录
        
    Returns:
        训练结果字典
    """
    print("\n" + "=" * 70)
    print(f"训练 {model_name} - 电池 {battery_name}")
    print("=" * 70)
    
    # 1. 加载数据
    data_path = os.path.join('data/HUST data', f'{battery_name}.csv')
    if not os.path.exists(data_path):
        print(f"❌ 文件不存在: {data_path}")
        return None
    
    print(f"\n加载数据: {battery_name}")
    data_dict = load_single_hust_battery(
        data_path,
        train_ratio=config['data']['train_ratio'],
        normalize_target=config['data']['normalize_target']
    )
    
    print(f"  训练集: {data_dict['n_train']} 个样本")
    print(f"  测试集: {data_dict['n_test']} 个样本")
    print(f"  特征数: {len(data_dict['feature_names'])}")
    
    # 2. 创建训练器
    trainer = ModelTrainer(config, device=device.type)
    
    # 3. 打印配置
    ConfigLoader.print_config(config)
    
    # 4. 训练
    history, results = trainer.train(data_dict, verbose=True)
    
    # 5. 保存模型
    battery_results_dir = os.path.join(results_dir, battery_name, model_name.lower())
    ensure_dir(battery_results_dir)
    
    model_path = os.path.join(battery_results_dir, f'{model_name.lower()}_model.pth')
    trainer.save_model(model_path)
    
    # 6. 保存结果
    results_path = os.path.join(battery_results_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump({
            'config': config,
            'history': history,
            'results': results,
            'battery_name': battery_name,
            'model_name': model_name,
            'data_info': {
                'n_train': data_dict['n_train'],
                'n_test': data_dict['n_test'],
                'n_features': len(data_dict['feature_names'])
            }
        }, f)
    print(f"✓ 结果已保存: {results_path}")
    
    # 7. 打印摘要
    print("\n" + "-" * 70)
    print(f"训练结果摘要 ({model_name} - {battery_name})")
    print("-" * 70)
    print(f"最终MAE:     {results['mae']*100:.4f}%")
    print(f"最终RMSE:    {results['rmse']*100:.4f}%")
    print(f"最佳Epoch:   {results['best_epoch']}")
    print(f"最佳MAE:     {results['best_mae']*100:.4f}%")
    print(f"最终训练损失: {results['final_train_loss']:.6f}")
    print("-" * 70)
    
    return {
        'model': model_name,
        'battery': battery_name,
        'results': results,
        'history': history
    }


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(
        description='HUST数据集基准模型统一训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 训练FNN
  python main_hust_baseline.py --model FNN --battery 1-1
  
  # 训练CNN (自定义学习率)
  python main_hust_baseline.py --model CNN --battery 1-1 --lr 0.0005
  
  # 训练多个电池
  python main_hust_baseline.py --model LSTM --batteries 1-1 1-2 1-3
        """
    )
    
    # 必要参数
    parser.add_argument('--model', type=str, required=True,
                        choices=['FNN', 'CNN', 'LSTM'],
                        help='选择模型 (FNN, CNN, LSTM)')
    
    # 电池选择
    battery_group = parser.add_mutually_exclusive_group(required=True)
    battery_group.add_argument('--battery', type=str,
                               help='单个电池 (例如: 1-1, 1-2, ..., 10-8)')
    battery_group.add_argument('--batteries', type=str, nargs='+',
                               help='多个电池列表 (例如: 1-1 1-2 1-3)')
    
    # 可选的配置覆盖
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率 (覆盖配置文件中的值)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='批量大小 (覆盖配置文件中的值)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数 (覆盖配置文件中的值)')
    parser.add_argument('--corr-threshold', type=float, default=None,
                        help='特征相关性阈值 (覆盖配置文件中的值)')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='不使用GPU')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device('cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载配置
    print(f"\n加载配置: {args.model}")
    try:
        config = ConfigLoader.load_model_config_by_name(args.model)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # 应用命令行参数覆盖
    if any([args.lr, args.batch_size, args.epochs, args.corr_threshold]):
        print(f"\n应用命令行参数覆盖:")
        config = override_config(config, args)
    
    # 确定要训练的电池列表
    if args.battery:
        batteries = [args.battery]
    else:
        batteries = args.batteries
    
    # 创建结果目录
    results_dir = 'results/results_hust'
    ensure_dir(results_dir)
    
    # 训练每个电池
    all_results = []
    for battery_name in batteries:
        result = train_battery(args.model, battery_name, config, device, results_dir)
        if result is not None:
            all_results.append(result)
    
    # 汇总结果
    if len(all_results) > 0:
        print("\n" + "=" * 70)
        print("训练汇总")
        print("=" * 70)
        
        print(f"\n模型: {args.model}")
        print(f"训练电池数: {len(all_results)}")
        print(f"\n{'电池':<12} {'MAE (%)':<12} {'RMSE (%)':<12} {'最佳Epoch':<15}")
        print("-" * 70)
        
        for result in all_results:
            mae = result['results']['mae'] * 100
            rmse = result['results']['rmse'] * 100
            best_epoch = result['results']['best_epoch']
            
            print(f"{result['battery']:<12} {mae:<12.4f} {rmse:<12.4f} {best_epoch:<15}")
        
        # 计算平均值
        if len(all_results) > 1:
            avg_mae = np.mean([r['results']['mae'] for r in all_results]) * 100
            avg_rmse = np.mean([r['results']['rmse'] for r in all_results]) * 100
            
            print("-" * 70)
            print(f"{'平均值':<12} {avg_mae:<12.4f} {avg_rmse:<12.4f}")
        
        print("=" * 70)
        print(f"\n✓ 所有结果已保存到: {results_dir}/")
    
    return all_results


if __name__ == "__main__":
    results = main()
