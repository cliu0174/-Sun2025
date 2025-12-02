%% 快速开始示例 - 电池SOH估计
% 这个脚本演示如何使用MATLAB版本进行快速训练和预测

clear; clc; close all;

fprintf('======================================================================\n');
fprintf('电池SOH估计 - 快速开始示例\n');
fprintf('======================================================================\n\n');

%% 步骤1: 测试数据加载
fprintf('[步骤 1/4] 测试数据加载功能...\n');

% 添加路径
addpath('data_loaders');
addpath('models');
addpath('utils');

% 测试加载单个电池
test_file = '../data/HUST data/1-1.csv';

if exist(test_file, 'file')
    fprintf('测试文件: %s\n', test_file);

    % 加载数据
    data = load_single_hust_battery(test_file, 0.75, true, false);

    fprintf('✅ 数据加载成功！\n');
    fprintf('   电池名称: %s\n', data.battery_name);
    fprintf('   训练样本: %d\n', data.n_train);
    fprintf('   测试样本: %d\n', data.n_test);
    fprintf('   特征数量: %d\n', length(data.feature_names));
    fprintf('   SOH范围: [%.4f, %.4f]\n\n', ...
        min(data.train_capacity), max(data.train_capacity));
else
    error('❌ 找不到测试文件: %s\n请检查数据路径！', test_file);
end

%% 步骤2: 测试数据清洗
fprintf('[步骤 2/4] 测试3-Sigma数据清洗...\n');

% 不清洗
data_raw = load_single_hust_battery(test_file, 1.0, true, false);
n_raw = data_raw.n_train;

% 清洗
data_cleaned = load_single_hust_battery(test_file, 1.0, true, true);
n_cleaned = data_cleaned.n_train;

fprintf('✅ 数据清洗测试完成！\n');
fprintf('   原始样本: %d\n', n_raw);
fprintf('   清洗后: %d\n', n_cleaned);
fprintf('   删除率: %.2f%%\n\n', (n_raw - n_cleaned) / n_raw * 100);

%% 步骤3: 测试滑动窗口
fprintf('[步骤 3/4] 测试滑动窗口功能...\n');

window_size = 10;
[X, y, battery_ids, cycle_indices] = apply_windowing_with_metadata(...
    data.train_features, data.train_capacity, window_size, data.battery_name, 'many_to_one');

fprintf('✅ 滑动窗口创建成功！\n');
fprintf('   原始样本: %d\n', size(data.train_features, 1));
fprintf('   窗口化后: %d\n', size(X, 1));
fprintf('   窗口大小: %d\n', window_size);
fprintf('   特征维度: [%d, %d, %d]\n\n', size(X, 1), size(X, 2), size(X, 3));

%% 步骤4: 测试模型创建
fprintf('[步骤 4/4] 测试模型创建...\n');

model_types = {'LSTM', 'GRU', 'CNN', 'CNN_LSTM'};

for i = 1:length(model_types)
    model_type = model_types{i};

    switch model_type
        case 'LSTM'
            layers = create_lstm_network(16, 64, 2, [32, 16], 0.2);
        case 'GRU'
            layers = create_gru_network(16, 64, 2, [32, 16], 0.2);
        case 'CNN'
            layers = create_cnn_network(16, 64, 3, [32, 16], 0.2);
        case 'CNN_LSTM'
            layers = create_cnn_lstm_network(16, [32, 64], 3, 2, 64, 2, [64], 0.2);
    end

    fprintf('   ✅ %s 网络创建成功 (%d 层)\n', model_type, length(layers));
end

fprintf('\n');

%% 步骤5: 测试物理约束损失
fprintf('[额外] 测试物理约束损失函数...\n');

% 创建测试数据
test_predictions = [0.92; 0.88; 0.915; 0.905; 0.875];
test_targets = test_predictions;  % 假设预测完美
test_battery_ids = {'1-1'; '1-1'; '1-1'; '1-1'; '1-1'};
test_cycle_indices = [50; 55; 60; 70; 80];

% 设置权重
weights = struct();
weights.base_weight = 1.0;
weights.monotonic_weight = 0.1;
weights.boundary_weight = 0.05;
weights.smoothness_weight = 0.0;
weights.monotonic_tolerance = 0.01;
weights.temporal_max_step = 20;
weights.temporal_decay_alpha = 0.2;

% 计算损失
[total_loss, loss_details] = compute_physics_loss(...
    test_predictions, test_targets, test_battery_ids, test_cycle_indices, weights);

fprintf('✅ 物理约束损失计算成功！\n');
fprintf('   总损失: %.6f\n', loss_details.total);
fprintf('   基础MSE: %.6f\n', loss_details.base);
fprintf('   单调性: %.6f\n', loss_details.monotonic);
fprintf('   边界约束: %.6f\n\n', loss_details.boundary);

%% 总结
fprintf('======================================================================\n');
fprintf('✅ 所有测试通过！系统运行正常。\n');
fprintf('======================================================================\n\n');

fprintf('下一步操作:\n');
fprintf('1. 运行完整训练: 执行 train_cross_battery\n');
fprintf('2. 查看文档: 打开 README.md\n');
fprintf('3. 修改配置: 编辑 train_cross_battery.m 中的 config 部分\n\n');

fprintf('提示: 如果需要快速测试，可以在 train_cross_battery.m 中:\n');
fprintf('      - 设置 config.max_epochs = 10 (减少训练轮数)\n');
fprintf('      - 只使用部分电池数据进行训练\n\n');
