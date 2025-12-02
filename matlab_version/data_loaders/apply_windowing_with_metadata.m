function [X, y, battery_ids, cycle_indices] = apply_windowing_with_metadata(features, targets, window_size, battery_id, mode)
% APPLY_WINDOWING_WITH_METADATA 应用滑动窗口并附加battery_id和cycle_idx
%
% 输入参数:
%   features: (total_cycles, feature_dim) 特征数组
%   targets: (total_cycles, 1) 目标值数组
%   window_size: 窗口大小
%   battery_id: 字符串，电池名称（如 "1-1"）
%   mode: 'many_to_one' 或 'many_to_many'
%
% 输出参数:
%   X: (N, window_size, feature_dim) 输入窗口
%   y: (N, 1) for many_to_one 或 (N, window_size) for many_to_many
%   battery_ids: (N, 1) cell array，每个窗口的电池ID
%   cycle_indices: (N, 1) 每个窗口对应的cycle索引

    if nargin < 5
        mode = 'many_to_one';
    end

    total_cycles = size(features, 1);
    feature_dim = size(features, 2);

    % 初始化输出数组
    num_windows = total_cycles - window_size + 1;
    X = zeros(num_windows, window_size, feature_dim);
    battery_ids = cell(num_windows, 1);
    cycle_indices = zeros(num_windows, 1);

    if strcmp(mode, 'many_to_one')
        y = zeros(num_windows, 1);
    elseif strcmp(mode, 'many_to_many')
        y = zeros(num_windows, window_size);
    else
        error('Unknown mode: %s. Must be ''many_to_one'' or ''many_to_many''', mode);
    end

    % 创建滑动窗口
    for i = 1:num_windows
        % 输入窗口
        X(i, :, :) = features(i:i+window_size-1, :);

        if strcmp(mode, 'many_to_one')
            % Many-to-One: 预测窗口最后一个点的目标值
            y(i) = targets(i+window_size-1);
            % cycle_idx = 窗口最后一个点的原始cycle索引
            cycle_idx = i + window_size - 1;

        elseif strcmp(mode, 'many_to_many')
            % Many-to-Many: 预测整个窗口的目标值
            y(i, :) = targets(i:i+window_size-1)';
            % cycle_idx 定义为窗口的结束位置
            cycle_idx = i + window_size - 1;
        end

        % 附加元数据
        battery_ids{i} = battery_id;
        cycle_indices(i) = cycle_idx;
    end
end
