%% 验证所有模型配置与Python版本一致
% 读取所有6个模型的配置文件，对比关键参数

clear; clc;

addpath('models');

fprintf('======================================================================\n');
fprintf('配置文件参数验证\n');
fprintf('======================================================================\n\n');

models = {'lstm', 'gru', 'cnn', 'cnn_lstm', 'bilstm', 'bigru'};

fprintf('%-12s | %-6s | %-5s | %-7s | %-9s | %-6s | %-8s | %-7s\n', ...
    'Model', 'Epochs', 'Batch', 'LR', 'Scheduler', 'Window', 'Physics', 'Dropout');
fprintf('%s\n', repmat('-', 1, 90));

for i = 1:length(models)
    model_type = models{i};

    try
        % 加载配置
        config = ConfigLoader.load_model_config(model_type);

        % 提取参数
        epochs = get_field(config, 'training.num_epochs', 'N/A');
        batch_size = get_field(config, 'training.batch_size', 'N/A');
        lr = get_field(config, 'training.learning_rate', 'N/A');

        % 学习率调度器
        if isfield(config, 'training') && isfield(config.training, 'scheduler')
            scheduler_enabled = config.training.scheduler.enabled;
            if scheduler_enabled
                scheduler_str = 'enabled';
            else
                scheduler_str = 'disabled';
            end
        else
            scheduler_str = 'N/A';
        end

        % 窗口大小
        window = get_field(config, 'data.window_size', 'N/A');

        % 物理约束
        if isfield(config, 'physics_constraints') && isfield(config.physics_constraints, 'enabled')
            physics_enabled = config.physics_constraints.enabled;
            if physics_enabled
                physics_str = 'true';
            else
                physics_str = 'false';
            end
        else
            physics_str = 'N/A';
        end

        % Dropout率
        dropout = get_field(config, 'architecture.dropout_rate', 'N/A');

        % 格式化学习率
        if isnumeric(lr)
            if lr < 0.001
                lr_str = sprintf('%.5f', lr);
            else
                lr_str = sprintf('%.4f', lr);
            end
        else
            lr_str = lr;
        end

        % 打印行
        fprintf('%-12s | %-6s | %-5s | %-7s | %-9s | %-6s | %-8s | %-7s\n', ...
            upper(model_type), ...
            num2str(epochs), ...
            num2str(batch_size), ...
            lr_str, ...
            scheduler_str, ...
            num2str(window), ...
            physics_str, ...
            num2str(dropout));

    catch ME
        fprintf('%-12s | ERROR: %s\n', upper(model_type), ME.message);
    end
end

fprintf('\n======================================================================\n');
fprintf('预期值对比\n');
fprintf('======================================================================\n\n');

fprintf('LSTM:\n');
fprintf('  ✓ epochs=200, batch=256, lr=0.001, scheduler=enabled, window=40, physics=true, dropout=0.4\n\n');

fprintf('GRU:\n');
fprintf('  ✓ epochs=200, batch=256, lr=0.001, scheduler=disabled, window=40, physics=false, dropout=0.4\n\n');

fprintf('CNN:\n');
fprintf('  ✓ epochs=200, batch=256, lr=0.0001, scheduler=disabled, window=N/A, physics=N/A, dropout=0.2\n\n');

fprintf('CNN-LSTM:\n');
fprintf('  ✓ epochs=200, batch=256, lr=0.004, scheduler=enabled, window=40, physics=false, dropout=0.4\n\n');

fprintf('BiLSTM:\n');
fprintf('  ✓ epochs=200, batch=256, lr=0.001, scheduler=disabled, window=40, physics=false, dropout=0.4\n\n');

fprintf('BiGRU:\n');
fprintf('  ✓ epochs=200, batch=256, lr=0.001, scheduler=disabled, window=30, physics=false, dropout=0.4\n\n');

%% 辅助函数
function value = get_field(struct_var, field_path, default_value)
    % 递归获取嵌套字段
    % 例如: get_field(config, 'training.num_epochs', 200)

    fields = strsplit(field_path, '.');
    current = struct_var;

    for i = 1:length(fields)
        field_name = fields{i};
        if isfield(current, field_name)
            current = current.(field_name);
        else
            value = default_value;
            return;
        end
    end

    value = current;
end
