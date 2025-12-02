%% 测试基本功能 - MATLAB版本
% 对应Python: 快速测试所有核心功能

clear; clc; close all;

fprintf('======================================================================\n');
fprintf('测试MATLAB版本基本功能\n');
fprintf('======================================================================\n\n');

%% 添加路径
addpath('models');
addpath('data_loaders');
addpath('utils');

%% ===== 测试1: ConfigLoader =====
fprintf('[测试 1/5] ConfigLoader 类...\n');
try
    % 加载LSTM配置
    config = ConfigLoader.load_model_config('lstm');
    fprintf('  ✅ 成功加载 LSTM 配置\n');
    fprintf('     - 隐藏层大小: %d\n', config.architecture.hidden_size);
    fprintf('     - 层数: %d\n', config.architecture.num_layers);
    fprintf('     - Batch大小: %d\n', config.training.batch_size);

    % 加载其他模型配置
    config_gru = ConfigLoader.load_model_config('gru');
    config_cnn = ConfigLoader.load_model_config('cnn');
    fprintf('  ✅ 成功加载 GRU, CNN 配置\n');
catch ME
    fprintf('  ❌ 错误: %s\n', ME.message);
end

fprintf('\n');

%% ===== 测试2: 数据加载 =====
fprintf('[测试 2/5] 数据加载函数...\n');
try
    test_file = '../data/HUST data/1-1.csv';

    if exist(test_file, 'file')
        % 测试load_single_hust_battery
        data = load_single_hust_battery(test_file, 0.75, true, false);
        fprintf('  ✅ 成功加载电池数据: %s\n', data.battery_name);
        fprintf('     - 训练样本: %d\n', data.n_train);
        fprintf('     - 测试样本: %d\n', data.n_test);
        fprintf('     - 特征数: %d\n', length(data.feature_names));

        % 测试滑动窗口
        [X, y, ids, cycles] = apply_windowing_with_metadata(...
            data.train_features, data.train_capacity, 10, data.battery_name, 'many_to_one');
        fprintf('  ✅ 成功应用滑动窗口\n');
        fprintf('     - 窗口化样本: %d\n', size(X, 1));
        fprintf('     - 窗口大小: %d\n', size(X, 2));

        % 测试3-Sigma清洗
        test_table = readtable(test_file);
        [cleaned, stats] = clean_3_sigma(test_table, false);
        fprintf('  ✅ 成功测试3-Sigma清洗\n');
        fprintf('     - 原始: %d, 清洗后: %d\n', stats.original_size, stats.final_size);
    else
        fprintf('  ⚠️  测试文件不存在: %s\n', test_file);
        fprintf('     请确保数据文件在正确位置\n');
    end
catch ME
    fprintf('  ❌ 错误: %s\n', ME.message);
end

fprintf('\n');

%% ===== 测试3: ModelFactory =====
fprintf('[测试 3/5] ModelFactory 类...\n');
try
    input_size = 16;

    % 测试LSTM
    layers_lstm = ModelFactory.create_model('lstm', input_size);
    fprintf('  ✅ 成功创建 LSTM 模型 (%d 层)\n', length(layers_lstm));

    % 测试GRU
    layers_gru = ModelFactory.create_model('gru', input_size);
    fprintf('  ✅ 成功创建 GRU 模型 (%d 层)\n', length(layers_gru));

    % 测试CNN
    layers_cnn = ModelFactory.create_model('cnn', input_size);
    fprintf('  ✅ 成功创建 CNN 模型 (%d 层)\n', length(layers_cnn));

    % 测试CNN-LSTM
    layers_cnn_lstm = ModelFactory.create_model('cnn_lstm', input_size);
    fprintf('  ✅ 成功创建 CNN-LSTM 模型 (%d 层)\n', length(layers_cnn_lstm));

    % 测试BiLSTM
    layers_bilstm = ModelFactory.create_model('bilstm', input_size);
    fprintf('  ✅ 成功创建 BiLSTM 模型 (%d 层)\n', length(layers_bilstm));

    % 测试BiGRU
    layers_bigru = ModelFactory.create_model('bigru', input_size);
    fprintf('  ✅ 成功创建 BiGRU 模型 (%d 层)\n', length(layers_bigru));

catch ME
    fprintf('  ❌ 错误: %s\n', ME.message);
end

fprintf('\n');

%% ===== 测试4: PhysicsLoss =====
fprintf('[测试 4/5] PhysicsLoss 类...\n');
try
    % 创建物理约束损失
    physics_loss = PhysicsLoss(...
        'base_loss_weight', 1.0, ...
        'monotonic_weight', 0.1, ...
        'boundary_weight', 0.05, ...
        'monotonic_tolerance', 0.01);

    % 测试数据（模拟单调下降的SOH）
    test_pred = [0.95; 0.93; 0.91; 0.89; 0.87];
    test_true = test_pred;
    test_ids = {'1-1'; '1-1'; '1-1'; '1-1'; '1-1'};
    test_cycles = [10; 20; 30; 40; 50];

    % 计算损失
    [loss, details] = physics_loss.compute(test_pred, test_true, test_ids, test_cycles);

    fprintf('  ✅ 成功计算物理约束损失\n');
    fprintf('     - 总损失: %.6f\n', details.total);
    fprintf('     - 基础MSE: %.6f\n', details.base);
    fprintf('     - 单调性: %.6f\n', details.monotonic);
    fprintf('     - 边界: %.6f\n', details.boundary);

    % 测试违反单调性的情况
    test_pred_bad = [0.95; 0.93; 0.96; 0.89; 0.87];  % 第3个值上升了
    [loss_bad, details_bad] = physics_loss.compute(test_pred_bad, test_true, test_ids, test_cycles);

    fprintf('  ✅ 测试违反单调性的情况\n');
    fprintf('     - 总损失: %.6f (应该更大)\n', details_bad.total);
    fprintf('     - 单调性损失: %.6f (应该 > 0)\n', details_bad.monotonic);

catch ME
    fprintf('  ❌ 错误: %s\n', ME.message);
end

fprintf('\n');

%% ===== 测试5: 配置文件兼容性 =====
fprintf('[测试 5/5] 配置文件兼容性...\n');
try
    % 列出所有可用配置
    config_dir = 'configs/models';
    json_files = dir(fullfile(config_dir, '*.json'));

    fprintf('  找到 %d 个配置文件:\n', length(json_files));
    for i = 1:min(5, length(json_files))
        [~, name, ~] = fileparts(json_files(i).name);
        fprintf('     - %s\n', name);
    end

    % 测试加载和保存
    config = ConfigLoader.load_model_config('lstm');
    temp_path = 'test_config_temp.json';
    ConfigLoader.save_config(config, temp_path);

    config_reloaded = ConfigLoader.load_config(temp_path);
    fprintf('  ✅ 配置保存和重新加载成功\n');

    % 清理临时文件
    if exist(temp_path, 'file')
        delete(temp_path);
    end

catch ME
    fprintf('  ❌ 错误: %s\n', ME.message);
end

fprintf('\n');

%% ===== 总结 =====
fprintf('======================================================================\n');
fprintf('✅ 所有基本功能测试完成！\n');
fprintf('======================================================================\n\n');

fprintf('下一步:\n');
fprintf('1. 运行完整训练: train_cross_battery\n');
fprintf('2. 查看README: open(''README.md'')\n');
fprintf('3. 修改模型配置: 编辑 configs/models/*.json\n\n');

fprintf('提示:\n');
fprintf('- 如果测试失败，请检查:\n');
fprintf('  * MATLAB版本 >= R2020b\n');
fprintf('  * Deep Learning Toolbox已安装\n');
fprintf('  * 数据文件路径正确\n\n');
