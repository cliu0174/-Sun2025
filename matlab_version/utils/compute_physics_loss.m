function [total_loss, loss_details] = compute_physics_loss(predictions, targets, battery_ids, cycle_indices, weights)
% COMPUTE_PHYSICS_LOSS 计算物理约束损失
%
% 输入参数:
%   predictions: (N, 1) 预测的SOH值
%   targets: (N, 1) 真实的SOH值
%   battery_ids: (N, 1) cell array，电池ID
%   cycle_indices: (N, 1) cycle索引
%   weights: 结构体，包含各项损失权重
%       - base_weight: 基础MSE损失权重 (默认1.0)
%       - monotonic_weight: 单调性约束权重 (默认0.1)
%       - boundary_weight: 边界约束权重 (默认0.05)
%       - smoothness_weight: 平滑性约束权重 (默认0.0)
%       - monotonic_tolerance: 单调性容忍度 (默认0.01)
%       - temporal_max_step: 最大时间步长 (默认20)
%       - temporal_decay_alpha: 时间衰减系数 (默认0.2)
%
% 输出参数:
%   total_loss: 总损失值
%   loss_details: 结构体，包含各项损失的详细信息

    % 设置默认权重
    if nargin < 5
        weights = struct();
    end

    if ~isfield(weights, 'base_weight'), weights.base_weight = 1.0; end
    if ~isfield(weights, 'monotonic_weight'), weights.monotonic_weight = 0.1; end
    if ~isfield(weights, 'boundary_weight'), weights.boundary_weight = 0.05; end
    if ~isfield(weights, 'smoothness_weight'), weights.smoothness_weight = 0.0; end
    if ~isfield(weights, 'monotonic_tolerance'), weights.monotonic_tolerance = 0.01; end
    if ~isfield(weights, 'temporal_max_step'), weights.temporal_max_step = 20; end
    if ~isfield(weights, 'temporal_decay_alpha'), weights.temporal_decay_alpha = 0.2; end

    % 1. 基础MSE损失
    base_loss = mean((predictions - targets).^2);

    % 2. 边界约束损失：SOH应该在[0, 1]范围内
    lower_violation = max(0, -predictions);  % 惩罚SOH < 0
    upper_violation = max(0, predictions - 1.0);  % 惩罚SOH > 1
    boundary_loss = mean(lower_violation + upper_violation);

    % 3. 单调性约束损失（带时间衰减）
    monotonic_loss = compute_monotonic_loss(predictions, battery_ids, cycle_indices, ...
        weights.monotonic_tolerance, weights.temporal_max_step, weights.temporal_decay_alpha);

    % 4. 平滑性约束损失（可选）
    if weights.smoothness_weight > 0
        smoothness_loss = compute_smoothness_loss(predictions, battery_ids, cycle_indices);
    else
        smoothness_loss = 0;
    end

    % 计算总损失
    total_loss = weights.base_weight * base_loss + ...
                 weights.monotonic_weight * monotonic_loss + ...
                 weights.boundary_weight * boundary_loss + ...
                 weights.smoothness_weight * smoothness_loss;

    % 返回详细损失信息
    loss_details = struct();
    loss_details.total = total_loss;
    loss_details.base = base_loss;
    loss_details.monotonic = monotonic_loss;
    loss_details.boundary = boundary_loss;
    loss_details.smoothness = smoothness_loss;
end


function mono_loss = compute_monotonic_loss(predictions, battery_ids, cycle_indices, ...
    tolerance, max_step, decay_alpha)
% 计算单调性约束损失（软约束 + 时间衰减）

    unique_batteries = unique(battery_ids);
    total_loss = 0;
    num_pairs = 0;

    for b = 1:length(unique_batteries)
        battery_id = unique_batteries{b};

        % 找出该电池的所有样本
        battery_mask = strcmp(battery_ids, battery_id);
        battery_preds = predictions(battery_mask);
        battery_cycles = cycle_indices(battery_mask);

        if length(battery_preds) < 2
            continue;
        end

        % 按cycle索引排序
        [sorted_cycles, sort_idx] = sort(battery_cycles);
        sorted_preds = battery_preds(sort_idx);

        n = length(sorted_preds);

        % 对所有配对计算违反量
        for i = 1:n-1
            for j = i+1:n
                cycle_diff = sorted_cycles(j) - sorted_cycles(i);

                % 只考虑时间间隔 <= max_step 的配对
                if cycle_diff > max_step
                    continue;
                end

                % 预测值差异（j应该小于等于i，因为SOH下降）
                pred_diff = sorted_preds(j) - sorted_preds(i);

                % 计算违反量（如果上升超过tolerance，则有违反）
                violation = max(0, pred_diff - tolerance);

                % 时间衰减权重（指数衰减）
                weight = exp(-decay_alpha * cycle_diff);

                % 累加损失
                total_loss = total_loss + weight * violation;
                num_pairs = num_pairs + 1;
            end
        end
    end

    % 归一化
    if num_pairs > 0
        mono_loss = total_loss / num_pairs;
    else
        mono_loss = 0;
    end
end


function smooth_loss = compute_smoothness_loss(predictions, battery_ids, cycle_indices)
% 计算平滑性约束损失（二阶差分）

    unique_batteries = unique(battery_ids);
    total_loss = 0;
    num_valid = 0;

    for b = 1:length(unique_batteries)
        battery_id = unique_batteries{b};

        % 找出该电池的所有样本
        battery_mask = strcmp(battery_ids, battery_id);
        battery_preds = predictions(battery_mask);
        battery_cycles = cycle_indices(battery_mask);

        if length(battery_preds) < 3
            continue;
        end

        % 按cycle索引排序
        [~, sort_idx] = sort(battery_cycles);
        sorted_preds = battery_preds(sort_idx);

        % 计算一阶差分
        first_diff = diff(sorted_preds);

        % 计算二阶差分
        second_diff = diff(first_diff);

        % 累加平方和
        total_loss = total_loss + sum(second_diff.^2);
        num_valid = num_valid + length(second_diff);
    end

    % 归一化
    if num_valid > 0
        smooth_loss = total_loss / num_valid;
    else
        smooth_loss = 0;
    end
end
