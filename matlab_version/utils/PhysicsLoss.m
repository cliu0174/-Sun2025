classdef PhysicsLoss
    % PHYSICSLOSS 物理约束损失函数
    % 对应Python: models/physics_loss.py::PhysicsConstrainedLoss
    %
    % 总损失:
    % L = w_base * L_MSE + w_mono * L_monotonic + w_bound * L_boundary + w_smooth * L_smoothness
    %
    % 用法:
    %   loss_fn = PhysicsLoss();
    %   [total_loss, details] = loss_fn.compute(predictions, targets, battery_ids, cycle_indices);

    properties
        % 损失权重（对应Python的__init__参数）
        base_loss_weight = 1.0
        monotonic_weight = 0.1
        boundary_weight = 0.05
        smoothness_weight = 0.0

        % 软约束参数
        monotonic_tolerance = 0.01

        % 时间衰减参数
        temporal_decay_enabled = true
        temporal_max_step = 20
        temporal_decay_type = 'exp'  % 'exp', 'linear', 'inverse'
        temporal_decay_alpha = 0.2

        % 调试参数
        verbose = false
    end

    methods
        function obj = PhysicsLoss(varargin)
            % 构造函数 - 初始化物理约束损失
            % 对应Python: PhysicsConstrainedLoss.__init__
            %
            % 可选参数（成对）:
            %   'base_loss_weight', 1.0
            %   'monotonic_weight', 0.1
            %   ...

            for i = 1:2:length(varargin)
                param_name = varargin{i};
                param_value = varargin{i+1};
                if isprop(obj, param_name)
                    obj.(param_name) = param_value;
                end
            end
        end

        function [total_loss, loss_details] = compute(obj, predictions, targets, battery_ids, cycle_indices)
            % 计算总损失
            % 对应Python: PhysicsConstrainedLoss.forward
            %
            % Args:
            %   predictions: (N, 1) 预测值
            %   targets: (N, 1) 真实值
            %   battery_ids: (N, 1) cell array，电池ID
            %   cycle_indices: (N, 1) cycle索引
            %
            % Returns:
            %   total_loss: 总损失值
            %   loss_details: 结构体，包含各项损失

            % 1. 基础MSE损失（对应Python: self.mse_loss）
            base_loss = mean((predictions - targets).^2);

            % 2. 边界约束损失（对应Python: self.boundary_loss）
            bound_loss = obj.compute_boundary_loss(predictions);

            % 3. 单调性约束损失（对应Python: self.monotonic_loss）
            if ~isempty(battery_ids) && ~isempty(cycle_indices)
                mono_loss = obj.compute_monotonic_loss(predictions, battery_ids, cycle_indices);
            else
                mono_loss = 0;
            end

            % 4. 平滑性约束损失（对应Python: self.smoothness_loss）
            if obj.smoothness_weight > 0 && ~isempty(battery_ids) && ~isempty(cycle_indices)
                smooth_loss = obj.compute_smoothness_loss(predictions, battery_ids, cycle_indices);
            else
                smooth_loss = 0;
            end

            % 计算总损失（对应Python的total_loss计算）
            total_loss = obj.base_loss_weight * base_loss + ...
                        obj.monotonic_weight * mono_loss + ...
                        obj.boundary_weight * bound_loss + ...
                        obj.smoothness_weight * smooth_loss;

            % 记录详细损失（对应Python: self.loss_details）
            loss_details = struct();
            loss_details.total = total_loss;
            loss_details.base = base_loss;
            loss_details.monotonic = mono_loss;
            loss_details.boundary = bound_loss;
            loss_details.smoothness = smooth_loss;
        end

        function loss = compute_boundary_loss(obj, predictions)
            % 边界约束：SOH应在[0, 1]范围内
            % 对应Python: PhysicsConstrainedLoss.boundary_loss

            % 惩罚SOH < 0
            lower_violation = max(0, -predictions);

            % 惩罚SOH > 1
            upper_violation = max(0, predictions - 1.0);

            loss = mean(lower_violation + upper_violation);
        end

        function loss = compute_monotonic_loss(obj, predictions, battery_ids, cycle_indices)
            % 软单调性约束 + 时间衰减权重
            % 对应Python: PhysicsConstrainedLoss.monotonic_loss

            total_loss = 0;
            num_pairs = 0;

            % 获取所有唯一的电池ID（对应Python: unique_batteries = set(battery_ids)）
            unique_batteries = unique(battery_ids, 'stable');

            if obj.verbose
                fprintf('\n[MonotonicLoss] Batch内有 %d 个不同电池\n', length(unique_batteries));
            end

            % 对每个电池分别处理（对应Python: for battery_id in unique_batteries）
            for b = 1:length(unique_batteries)
                battery_id = unique_batteries{b};

                % 找出该电池的所有样本索引
                mask = strcmp(battery_ids, battery_id);
                indices = find(mask);

                if length(indices) < 2
                    if obj.verbose
                        fprintf('  电池 %s: 只有 %d 个样本，跳过\n', battery_id, length(indices));
                    end
                    continue;
                end

                % 提取该电池的预测值和cycle索引
                battery_preds = predictions(indices);
                battery_cycles = cycle_indices(indices);

                % 按cycle_idx排序（对应Python: sorted_indices = torch.argsort）
                [sorted_cycles, sort_idx] = sort(battery_cycles);
                sorted_preds = battery_preds(sort_idx);

                if obj.verbose
                    fprintf('  电池 %s: %d 个样本\n', battery_id, length(indices));
                end

                % 对所有可能的配对应用约束（对应Python的矢量化版本）
                n = length(sorted_preds);

                for i = 1:n-1
                    for j = i+1:n
                        cycle_diff = sorted_cycles(j) - sorted_cycles(i);

                        % 只考虑时间间隔 <= max_step的配对
                        if cycle_diff > obj.temporal_max_step
                            continue;
                        end

                        % 预测值差异
                        pred_diff = sorted_preds(j) - sorted_preds(i);

                        % 计算时间衰减权重
                        if obj.temporal_decay_enabled
                            weight = obj.compute_decay_weight(cycle_diff);
                        else
                            weight = 1.0;
                        end

                        % 计算违反量（SOH应该下降）
                        violation = max(0, pred_diff - obj.monotonic_tolerance);

                        % 累加损失
                        total_loss = total_loss + weight * violation;
                        num_pairs = num_pairs + 1;
                    end
                end
            end

            % 归一化
            if num_pairs > 0
                loss = total_loss / num_pairs;
                if obj.verbose
                    fprintf('  总配对数: %d, 归一化损失: %.6f\n', num_pairs, loss);
                end
            else
                loss = 0;
                if obj.verbose
                    fprintf('  无有效配对\n');
                end
            end
        end

        function weight = compute_decay_weight(obj, k)
            % 计算时间衰减权重
            % 对应Python: PhysicsConstrainedLoss.compute_decay_weight

            switch obj.temporal_decay_type
                case 'exp'
                    % 指数衰减
                    weight = exp(-obj.temporal_decay_alpha * k);
                case 'linear'
                    % 线性衰减
                    weight = max(0, 1.0 - obj.temporal_decay_alpha * k);
                case 'inverse'
                    % 倒数衰减
                    weight = 1.0 / k;
                otherwise
                    weight = 1.0;
            end
        end

        function loss = compute_smoothness_loss(obj, predictions, battery_ids, cycle_indices)
            % 平滑性约束：SOH变化应平滑（二阶差分小）
            % 对应Python: PhysicsConstrainedLoss.smoothness_loss

            total_loss = 0;
            num_valid = 0;

            % 获取唯一电池
            unique_batteries = unique(battery_ids, 'stable');

            for b = 1:length(unique_batteries)
                battery_id = unique_batteries{b};

                % 找出该电池的所有样本
                mask = strcmp(battery_ids, battery_id);
                indices = find(mask);

                if length(indices) < 3
                    % 需要至少3个点才能计算二阶差分
                    continue;
                end

                % 提取并排序
                battery_preds = predictions(indices);
                battery_cycles = cycle_indices(indices);

                [~, sort_idx] = sort(battery_cycles);
                sorted_preds = battery_preds(sort_idx);

                % 计算一阶差分（对应Python: first_diff = sorted_preds[1:] - sorted_preds[:-1]）
                first_diff = diff(sorted_preds);

                % 计算二阶差分（对应Python: second_diff = first_diff[1:] - first_diff[:-1]）
                second_diff = diff(first_diff);

                % 二阶差分的平方和
                total_loss = total_loss + sum(second_diff.^2);
                num_valid = num_valid + length(second_diff);
            end

            if num_valid > 0
                loss = total_loss / num_valid;
            else
                loss = 0;
            end
        end
    end
end
