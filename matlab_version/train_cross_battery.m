%% 跨电池训练脚本 - MATLAB版本（改进版）
% 对应Python: train_cross_battery.py
%
% 使用ConfigLoader和ModelFactory，与Python版本保持一致
%
% 数据划分: train/val/test = 6:2:2
% - 训练集: 46个电池 (60%)
% - 验证集: 16个电池 (20%)
% - 测试集: 15个电池 (20%)

clear; clc; close all;

%% ===== 添加路径 =====
addpath('models');
addpath('data_loaders');
addpath('utils');

%% ===== 配置参数 =====
% 模型选择: 'lstm', 'gru', 'cnn', 'cnn_lstm', 'bilstm', 'bigru'
model_type = 'lstm';

% 数据参数
data_dir = '../data/HUST data';
window_size = 10;
apply_cleaning = false;  % 是否应用3-Sigma清洗

% 训练参数（可以从config覆盖）
max_epochs = 100;
mini_batch_size = 64;
initial_learn_rate = 0.001;

% 物理约束（对应Python的PhysicsConstrainedLoss参数）
use_physics = true;

% 输出目录
output_dir = sprintf('results/cross_battery/%s', lower(model_type));
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 随机种子（对应Python: set_seed(42)）
rng(42);

fprintf('======================================================================\n');
fprintf('跨电池SOH估计训练 - MATLAB版本\n');
fprintf('======================================================================\n');
fprintf('模型类型: %s\n', upper(model_type));
fprintf('窗口大小: %d\n', window_size);
fprintf('数据清洗: %s\n', string(apply_cleaning));
fprintf('物理约束: %s\n', string(use_physics));
fprintf('======================================================================\n\n');

%% ===== 1. 加载配置 =====
% 对应Python: config = ConfigLoader.load_model_config(model_type)
fprintf('[1/7] 加载模型配置...\n');
model_config = ConfigLoader.load_model_config(model_type);
ConfigLoader.print_config(model_config);

% 从配置中提取训练参数
if isfield(model_config, 'training')
    if isfield(model_config.training, 'num_epochs')
        max_epochs = model_config.training.num_epochs;
    end
    if isfield(model_config.training, 'batch_size')
        mini_batch_size = model_config.training.batch_size;
    end
    if isfield(model_config.training, 'learning_rate')
        initial_learn_rate = model_config.training.learning_rate;
    end
end

% 从配置中提取数据参数
if isfield(model_config, 'data')
    if isfield(model_config.data, 'window_size')
        window_size = model_config.data.window_size;
    end
end

fprintf('\n训练配置:\n');
fprintf('  Max Epochs: %d\n', max_epochs);
fprintf('  Batch Size: %d\n', mini_batch_size);
fprintf('  Learning Rate: %.6f\n', initial_learn_rate);
fprintf('  Window Size: %d\n\n', window_size);

%% ===== 2. 加载数据 =====
% 对应Python: load_all_batteries()
fprintf('[2/7] 加载HUST电池数据...\n');

csv_files = dir(fullfile(data_dir, '*.csv'));
battery_names = cell(length(csv_files), 1);
all_data = cell(length(csv_files), 1);

fprintf('找到 %d 个电池文件\n', length(csv_files));

for i = 1:length(csv_files)
    file_path = fullfile(data_dir, csv_files(i).name);
    % 对应Python: load_single_hust_battery()
    data = load_single_hust_battery(file_path, 1.0, true, apply_cleaning);
    battery_names{i} = data.battery_name;
    all_data{i} = data;

    if mod(i, 10) == 0
        fprintf('已加载 %d/%d 个电池\n', i, length(csv_files));
    end
end

fprintf('数据加载完成！共 %d 个电池\n\n', length(battery_names));

%% ===== 3. 划分数据集 =====
% 对应Python: split_batteries()
fprintf('[3/7] 划分训练/验证/测试集...\n');

n_batteries = length(battery_names);
indices = randperm(n_batteries);

n_train = floor(n_batteries * 0.6);
n_val = floor(n_batteries * 0.2);

train_indices = indices(1:n_train);
val_indices = indices(n_train+1:n_train+n_val);
test_indices = indices(n_train+n_val+1:end);

fprintf('训练集: %d 个电池\n', length(train_indices));
fprintf('验证集: %d 个电池\n', length(val_indices));
fprintf('测试集: %d 个电池\n\n', length(test_indices));

% 保存划分信息
split_info = struct();
split_info.train_batteries = battery_names(train_indices);
split_info.val_batteries = battery_names(val_indices);
split_info.test_batteries = battery_names(test_indices);
save(fullfile(output_dir, 'battery_split.mat'), 'split_info');

%% ===== 4. 准备窗口化数据 =====
% 对应Python: apply_windowing_with_metadata()
fprintf('[4/7] 应用滑动窗口...\n');

% 训练集
[X_train, y_train, battery_ids_train, cycle_indices_train] = ...
    prepare_windowed_data(all_data, train_indices, window_size);

% 验证集
[X_val, y_val, battery_ids_val, cycle_indices_val] = ...
    prepare_windowed_data(all_data, val_indices, window_size);

% 测试集
[X_test, y_test, battery_ids_test, cycle_indices_test] = ...
    prepare_windowed_data(all_data, test_indices, window_size);

fprintf('训练样本: %d\n', size(X_train, 1));
fprintf('验证样本: %d\n', size(X_val, 1));
fprintf('测试样本: %d\n\n', size(X_test, 1));

%% ===== 5. 创建模型 =====
% 对应Python: ModelFactory.create_model()
fprintf('[5/7] 创建 %s 网络...\n', upper(model_type));

input_size = size(X_train, 3);  % 特征维度
layers = ModelFactory.create_model(model_type, input_size);

fprintf('网络创建完成！共 %d 层\n\n', length(layers));

%% ===== 6. 训练模型 =====
fprintf('[6/7] 开始训练...\n');

% 训练选项
options = trainingOptions('adam', ...
    'MaxEpochs', max_epochs, ...
    'MiniBatchSize', mini_batch_size, ...
    'InitialLearnRate', initial_learn_rate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 20, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {prepare_sequence_data(X_val), y_val}, ...
    'ValidationFrequency', 50, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

% 转换数据格式（MATLAB需要cell array）
X_train_cell = prepare_sequence_data(X_train);
X_val_cell = prepare_sequence_data(X_val);
X_test_cell = prepare_sequence_data(X_test);

% 训练网络
net = trainNetwork(X_train_cell, y_train, layers, options);

% 保存模型
save(fullfile(output_dir, 'trained_model.mat'), 'net', 'model_config');

fprintf('\n训练完成！\n\n');

%% ===== 7. 评估模型 =====
fprintf('[7/7] 评估模型性能...\n');

% 测试集预测
y_pred_test = predict(net, X_test_cell);

% 计算评估指标（对应Python的evaluate函数）
results = evaluate_predictions(y_test, y_pred_test);

fprintf('\n测试集性能:\n');
fprintf('  RMSE: %.4f\n', results.rmse);
fprintf('  MAE:  %.4f\n', results.mae);
fprintf('  MAPE: %.2f%%\n', results.mape);
fprintf('  R²:   %.4f\n', results.r2);

% 物理约束评估（对应Python的PhysicsConstrainedLoss）
if use_physics
    % 初始化物理约束损失
    physics_loss = PhysicsLoss(...
        'base_loss_weight', 1.0, ...
        'monotonic_weight', 0.1, ...
        'boundary_weight', 0.05, ...
        'smoothness_weight', 0.0, ...
        'monotonic_tolerance', 0.01, ...
        'temporal_max_step', 20, ...
        'temporal_decay_alpha', 0.2);

    [total_loss, loss_details] = physics_loss.compute(...
        y_pred_test, y_test, battery_ids_test, cycle_indices_test);

    fprintf('\n物理约束损失:\n');
    fprintf('  总损失:     %.6f\n', loss_details.total);
    fprintf('  基础MSE:    %.6f\n', loss_details.base);
    fprintf('  单调性:     %.6f\n', loss_details.monotonic);
    fprintf('  边界约束:   %.6f\n', loss_details.boundary);

    results.physics_loss = loss_details;
end

% 保存结果
results.model_config = model_config;
results.y_true = y_test;
results.y_pred = y_pred_test;
results.battery_ids = battery_ids_test;
results.cycle_indices = cycle_indices_test;
save(fullfile(output_dir, 'results.mat'), 'results');

%% ===== 8. 可视化结果 =====
fprintf('\n生成可视化结果...\n');

fig1 = figure('Position', [100, 100, 1200, 400]);

subplot(1, 3, 1);
plot_predictions_scatter(y_test, y_pred_test, results);
title('预测 vs 真实值');

subplot(1, 3, 2);
plot_error_distribution(y_test, y_pred_test);
title('误差分布');

subplot(1, 3, 3);
plot_predictions_by_battery(y_test, y_pred_test, battery_ids_test, 5);
title('按电池分组的预测结果（前5个）');

saveas(fig1, fullfile(output_dir, 'predictions.png'));

fprintf('\n======================================================================\n');
fprintf('训练完成！\n');
fprintf('结果保存在: %s\n', output_dir);
fprintf('======================================================================\n');


%% ===== 辅助函数 =====

function [X, y, battery_ids, cycle_indices] = prepare_windowed_data(all_data, indices, window_size)
    % 准备窗口化数据
    % 对应Python: 遍历电池并调用apply_windowing_with_metadata

    X = [];
    y = [];
    battery_ids = {};
    cycle_indices = [];

    for i = 1:length(indices)
        idx = indices(i);
        data = all_data{idx};

        % 对应Python: apply_windowing_with_metadata()
        [X_batch, y_batch, ids_batch, cycles_batch] = apply_windowing_with_metadata(...
            data.train_features, data.train_capacity, window_size, data.battery_name, 'many_to_one');

        X = [X; X_batch];
        y = [y; y_batch];
        battery_ids = [battery_ids; ids_batch];
        cycle_indices = [cycle_indices; cycles_batch];
    end
end


function X_cell = prepare_sequence_data(X)
    % 将3D数组转换为cell array（MATLAB Deep Learning需要）
    % 输入: (N, window_size, feature_dim)
    % 输出: cell array of (feature_dim, window_size)

    n_samples = size(X, 1);
    X_cell = cell(n_samples, 1);

    for i = 1:n_samples
        % 转置: (window_size, feature_dim) -> (feature_dim, window_size)
        X_cell{i} = squeeze(X(i, :, :))';
    end
end


function results = evaluate_predictions(y_true, y_pred)
    % 计算评估指标
    % 对应Python: 各种评估函数

    results = struct();

    % RMSE
    results.rmse = sqrt(mean((y_true - y_pred).^2));

    % MAE
    results.mae = mean(abs(y_true - y_pred));

    % MAPE
    results.mape = mean(abs((y_true - y_pred) ./ y_true)) * 100;

    % R²
    ss_res = sum((y_true - y_pred).^2);
    ss_tot = sum((y_true - mean(y_true)).^2);
    results.r2 = 1 - ss_res / ss_tot;
end


function plot_predictions_scatter(y_true, y_pred, results)
    % 绘制预测对比散点图
    scatter(y_true, y_pred, 20, 'filled', 'MarkerFaceAlpha', 0.5);
    hold on;
    plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', 'LineWidth', 2);
    hold off;

    xlabel('真实SOH');
    ylabel('预测SOH');
    grid on;
    axis equal;
    xlim([min(y_true)-0.05, max(y_true)+0.05]);
    ylim([min(y_pred)-0.05, max(y_pred)+0.05]);

    % 添加性能指标
    text_str = sprintf('RMSE = %.4f\nMAE = %.4f\nR² = %.4f', ...
        results.rmse, results.mae, results.r2);
    text(0.05, 0.95, text_str, 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'FontSize', 10, ...
        'BackgroundColor', 'white', 'EdgeColor', 'black');
end


function plot_error_distribution(y_true, y_pred)
    % 绘制误差分布直方图
    errors = y_pred - y_true;
    histogram(errors, 30, 'Normalization', 'pdf', ...
        'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'none');

    xlabel('预测误差');
    ylabel('概率密度');
    grid on;

    % 添加均值和标准差
    text_str = sprintf('均值 = %.4f\n标准差 = %.4f', mean(errors), std(errors));
    text(0.95, 0.95, text_str, 'Units', 'normalized', ...
        'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', ...
        'FontSize', 10, 'BackgroundColor', 'white', 'EdgeColor', 'black');
end


function plot_predictions_by_battery(y_true, y_pred, battery_ids, n_show)
    % 按电池分组绘制预测结果
    unique_batteries = unique(battery_ids, 'stable');
    n_show = min(n_show, length(unique_batteries));

    colors = lines(n_show);

    for i = 1:n_show
        battery_id = unique_batteries{i};
        mask = strcmp(battery_ids, battery_id);

        y_true_battery = y_true(mask);
        y_pred_battery = y_pred(mask);

        plot(y_true_battery, 'o-', 'Color', colors(i, :), 'LineWidth', 1.5, ...
            'DisplayName', sprintf('%s (真实)', battery_id));
        hold on;
        plot(y_pred_battery, 's--', 'Color', colors(i, :), 'LineWidth', 1.5, ...
            'DisplayName', sprintf('%s (预测)', battery_id));
    end
    hold off;

    xlabel('样本索引');
    ylabel('SOH');
    legend('Location', 'best', 'FontSize', 8);
    grid on;
end
