"""
使用已训练模型对目标电池进行SOH预测

功能：
1. 加载已保存的模型参数
2. 对指定电池（如9-5）进行SOH预测
3. 可视化预测结果
"""

import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.model_factory import ModelFactory, ConfigLoader
from data_loader import load_battery_data
import joblib

# ============================================================================
#                              【配置区域】
# ============================================================================

# 目标电池ID
TARGET_BATTERY = '9-5'

# 已保存模型的目录（从 compare_models.py 训练后生成的目录）
# 示例: 'results/model_comparison_20260203_152341/saved_models'
SAVED_MODELS_DIR = 'results/model_comparison_20260203_152341/saved_models'

# 要使用的模型列表（确保这些模型已经训练并保存）
MODELS_TO_USE = [
    'cnn_lstm',
    'pi_cnn_lstm',
    'lstm',
    'xgboost_simple'
]

# 结果保存目录
OUTPUT_DIR = 'inference_results'

# 数据处理参数（应与训练时一致）
WINDOW_SIZE = 40
APPLY_CLEANING = False

# ============================================================================
#                           【配置区域结束】
# ============================================================================


def load_saved_model(model_type, saved_models_dir, device='cpu'):
    """
    加载已保存的模型参数。

    Args:
        model_type: 模型类型
        saved_models_dir: 模型保存目录
        device: 计算设备

    Returns:
        model: 加载了参数的模型
        config: 模型配置
    """
    model_path = os.path.join(saved_models_dir, f'{model_type}.pth')
    config_path = os.path.join(saved_models_dir, f'{model_type}_config.pkl')

    # 加载配置
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'rb') as f:
        saved_config = pickle.load(f)

    # 加载模型
    if model_type in ['xgboost_simple', 'xgboost_enhanced']:
        # XGBoost 模型
        xgb_model_path = model_path.replace('.pth', '.joblib')
        if not os.path.exists(xgb_model_path):
            raise FileNotFoundError(f"XGBoost模型文件不存在: {xgb_model_path}")

        xgb_model = joblib.load(xgb_model_path)

        # 创建模型包装器
        from models.xgboost_baseline import XGBoostSimple, XGBoostEnhanced

        if model_type == 'xgboost_simple':
            model = XGBoostSimple(input_size=saved_config['n_features'], window_size=WINDOW_SIZE)
        else:
            model = XGBoostEnhanced(input_size=saved_config['n_features'], window_size=WINDOW_SIZE)

        model.model = xgb_model
        print(f"✓ 已加载XGBoost模型: {xgb_model_path}")

    else:
        # PyTorch 模型
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        checkpoint = torch.load(model_path, map_location=device)

        # 创建模型
        model = ModelFactory.create_model(
            model_type=model_type,
            input_size=saved_config['n_features']
        )

        # 加载参数
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        print(f"✓ 已加载PyTorch模型: {model_path}")

    return model, saved_config


def prepare_battery_data(battery_id, window_size=40, apply_cleaning=False):
    """
    准备目标电池的数据用于预测。

    Args:
        battery_id: 电池ID
        window_size: 滑动窗口大小
        apply_cleaning: 是否应用数据清洗

    Returns:
        windows: 滑动窗口特征 [N, window_size, n_features]
        soh_values: SOH真实值 [N]
        cycle_indices: 循环索引 [N]
    """
    # 加载电池数据
    battery_data = load_battery_data(battery_id, apply_cleaning=apply_cleaning)

    if battery_data is None:
        raise ValueError(f"无法加载电池 {battery_id} 的数据")

    features = battery_data['features']
    soh = battery_data['soh']
    n_samples = len(soh)

    print(f"\n电池 {battery_id} 数据统计:")
    print(f"  总样本数: {n_samples}")
    print(f"  特征维度: {features.shape[1]}")
    print(f"  SOH范围: [{soh.min():.4f}, {soh.max():.4f}]")

    # 创建滑动窗口
    windows = []
    soh_values = []
    cycle_indices = []

    for i in range(window_size, n_samples):
        window = features[i - window_size:i, :]
        windows.append(window)
        soh_values.append(soh[i])
        cycle_indices.append(i)

    windows = np.array(windows)  # [N, window_size, n_features]
    soh_values = np.array(soh_values)  # [N]
    cycle_indices = np.array(cycle_indices)  # [N]

    print(f"  滑动窗口数量: {len(windows)}")

    return windows, soh_values, cycle_indices, features.shape[1]


def predict_battery(model, model_type, windows, device='cpu'):
    """
    使用模型对电池数据进行预测。

    Args:
        model: 训练好的模型
        model_type: 模型类型
        windows: 滑动窗口数据 [N, window_size, n_features]
        device: 计算设备

    Returns:
        predictions: 预测的SOH值 [N]
    """
    model.eval()

    if model_type in ['xgboost_simple', 'xgboost_enhanced']:
        # XGBoost 预测
        with torch.no_grad():
            X = torch.FloatTensor(windows)
            predictions = model(X)  # 返回torch.Tensor
            predictions = predictions.cpu().numpy().flatten()
    else:
        # PyTorch 模型预测
        all_predictions = []

        with torch.no_grad():
            batch_size = 128
            n_samples = len(windows)

            for i in range(0, n_samples, batch_size):
                batch_windows = windows[i:i + batch_size]
                batch_tensor = torch.FloatTensor(batch_windows).to(device)

                # 预测
                batch_pred = model(batch_tensor)

                # 处理不同的输出格式
                if isinstance(batch_pred, tuple):
                    batch_pred = batch_pred[0]

                all_predictions.append(batch_pred.cpu().numpy())

        predictions = np.concatenate(all_predictions).flatten()

    return predictions


def visualize_predictions(battery_id, cycle_indices, true_soh, predictions_dict, save_dir):
    """
    可视化预测结果。

    Args:
        battery_id: 电池ID
        cycle_indices: 循环索引
        true_soh: 真实SOH值
        predictions_dict: {model_type: predictions} 预测结果字典
        save_dir: 保存目录
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 配色方案
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions_dict) + 1))

    # 子图1: 预测值对比
    ax1 = axes[0]
    ax1.plot(cycle_indices, true_soh, 'o-', color='black', linewidth=2.5,
             markersize=5, label='True SOH', alpha=0.8, zorder=100)

    for idx, (model_type, predictions) in enumerate(predictions_dict.items()):
        # 计算误差
        rmse = np.sqrt(np.mean((predictions - true_soh) ** 2))
        mae = np.mean(np.abs(predictions - true_soh))

        ax1.plot(cycle_indices, predictions, 'o-', color=colors[idx],
                linewidth=2, markersize=3,
                label=f'{model_type.upper()} (RMSE={rmse:.4f}, MAE={mae:.4f})',
                alpha=0.7)

    ax1.set_xlabel('Cycle Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('SOH', fontsize=12, fontweight='bold')
    ax1.set_title(f'SOH Prediction for Battery {battery_id}',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # 子图2: 预测误差对比
    ax2 = axes[1]
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2.5,
                label='Zero Error', alpha=0.8, zorder=100)

    for idx, (model_type, predictions) in enumerate(predictions_dict.items()):
        errors = predictions - true_soh
        rmse = np.sqrt(np.mean(errors ** 2))
        mae = np.mean(np.abs(errors))

        ax2.plot(cycle_indices, errors, 'o-', color=colors[idx],
                linewidth=2, markersize=3,
                label=f'{model_type.upper()} (RMSE={rmse:.4f}, MAE={mae:.4f})',
                alpha=0.7)

    ax2.set_xlabel('Cycle Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Prediction Error (Predicted - True)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Prediction Error for Battery {battery_id}',
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'battery_{battery_id}_predictions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ 预测结果图已保存: {save_path}")


def save_predictions(battery_id, cycle_indices, true_soh, predictions_dict, save_dir):
    """
    保存预测结果到文件。

    Args:
        battery_id: 电池ID
        cycle_indices: 循环索引
        true_soh: 真实SOH值
        predictions_dict: {model_type: predictions} 预测结果字典
        save_dir: 保存目录
    """
    # 创建结果字典
    results = {
        'battery_id': battery_id,
        'cycle_indices': cycle_indices,
        'true_soh': true_soh,
        'predictions': predictions_dict
    }

    # 计算所有模型的误差指标
    metrics = {}
    for model_type, predictions in predictions_dict.items():
        rmse = np.sqrt(np.mean((predictions - true_soh) ** 2))
        mae = np.mean(np.abs(predictions - true_soh))
        mape = np.mean(np.abs((predictions - true_soh) / true_soh)) * 100

        from sklearn.metrics import r2_score
        r2 = r2_score(true_soh, predictions)

        metrics[model_type] = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        }

    results['metrics'] = metrics

    # 保存为pickle
    pkl_path = os.path.join(save_dir, f'battery_{battery_id}_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"✓ 预测结果已保存: {pkl_path}")

    # 打印性能指标
    print(f"\n{'='*70}")
    print(f"电池 {battery_id} 预测性能汇总")
    print(f"{'='*70}")
    for model_type, model_metrics in metrics.items():
        print(f"\n{model_type.upper()}:")
        print(f"  RMSE: {model_metrics['RMSE']*100:.4f}%")
        print(f"  MAE:  {model_metrics['MAE']*100:.4f}%")
        print(f"  MAPE: {model_metrics['MAPE']:.4f}%")
        print(f"  R²:   {model_metrics['R2']:.6f}")


def main():
    print("=" * 70)
    print("目标电池SOH预测")
    print("=" * 70)
    print(f"目标电池: {TARGET_BATTERY}")
    print(f"模型目录: {SAVED_MODELS_DIR}")
    print(f"使用模型: {MODELS_TO_USE}")

    # 检查模型目录是否存在
    if not os.path.exists(SAVED_MODELS_DIR):
        print(f"\n错误: 模型目录不存在: {SAVED_MODELS_DIR}")
        print("请先运行 compare_models.py 训练模型并保存参数")
        return

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 确定计算设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"计算设备: {device}")

    # 准备目标电池数据
    print("\n" + "=" * 70)
    print("准备电池数据")
    print("=" * 70)

    try:
        windows, true_soh, cycle_indices, n_features = prepare_battery_data(
            TARGET_BATTERY,
            window_size=WINDOW_SIZE,
            apply_cleaning=APPLY_CLEANING
        )
    except Exception as e:
        print(f"错误: 无法加载电池数据: {e}")
        return

    # 加载模型并进行预测
    print("\n" + "=" * 70)
    print("加载模型并预测")
    print("=" * 70)

    predictions_dict = {}

    for model_type in MODELS_TO_USE:
        print(f"\n处理模型: {model_type.upper()}")

        try:
            # 加载模型
            model, config = load_saved_model(model_type, SAVED_MODELS_DIR, device)

            # 预测
            predictions = predict_battery(model, model_type, windows, device)
            predictions_dict[model_type] = predictions

            # 计算误差
            rmse = np.sqrt(np.mean((predictions - true_soh) ** 2))
            mae = np.mean(np.abs(predictions - true_soh))

            print(f"  预测完成: RMSE={rmse:.6f}, MAE={mae:.6f}")

        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()

    if not predictions_dict:
        print("\n错误: 没有成功加载任何模型")
        return

    # 可视化预测结果
    print("\n" + "=" * 70)
    print("生成可视化结果")
    print("=" * 70)

    visualize_predictions(TARGET_BATTERY, cycle_indices, true_soh, predictions_dict, OUTPUT_DIR)

    # 保存预测结果
    save_predictions(TARGET_BATTERY, cycle_indices, true_soh, predictions_dict, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("预测完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
