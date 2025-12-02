classdef ConfigLoader
    % CONFIG配置文件加载器（对应Python的ConfigLoader类）
    %
    % 用法:
    %   config = ConfigLoader.load_config('configs/models/lstm_config.json');
    %   config = ConfigLoader.load_model_config('lstm');

    methods (Static)
        function config = load_config(config_path)
            % 加载JSON配置文件
            %
            % Args:
            %   config_path: 配置文件路径
            %
            % Returns:
            %   config: 配置结构体

            if ~exist(config_path, 'file')
                error('Configuration file not found: %s', config_path);
            end

            % MATLAB R2020b+ 支持 jsondecode
            fid = fopen(config_path, 'r', 'n', 'UTF-8');
            raw = fread(fid, inf);
            str = char(raw');
            fclose(fid);

            config = jsondecode(str);
        end

        function config = load_model_config(model_type, config_dir)
            % 通过模型类型加载配置
            %
            % Args:
            %   model_type: 模型类型 ('lstm', 'gru', 'cnn', etc.)
            %   config_dir: 配置目录（可选，默认'configs/models'）
            %
            % Returns:
            %   config: 配置结构体

            if nargin < 2
                config_dir = 'configs/models';
            end

            model_type = lower(model_type);
            config_path = fullfile(config_dir, sprintf('%s_config.json', model_type));
            config = ConfigLoader.load_config(config_path);
        end

        function save_config(config, save_path)
            % 保存配置到文件
            %
            % Args:
            %   config: 配置结构体
            %   save_path: 保存路径

            [save_dir, ~, ~] = fileparts(save_path);
            if ~isempty(save_dir) && ~exist(save_dir, 'dir')
                mkdir(save_dir);
            end

            % 使用jsonencode (MATLAB R2016b+)
            json_str = jsonencode(config);

            % 格式化JSON（简单版本）
            json_str = strrep(json_str, ',', sprintf(',\n  '));
            json_str = strrep(json_str, '{', sprintf('{\n  '));
            json_str = strrep(json_str, '}', sprintf('\n}'));

            fid = fopen(save_path, 'w', 'n', 'UTF-8');
            fprintf(fid, '%s', json_str);
            fclose(fid);

            fprintf('Configuration saved to: %s\n', save_path);
        end

        function print_config(config)
            % 打印配置信息
            %
            % Args:
            %   config: 配置结构体

            fprintf('\n%s\n', repmat('=', 1, 70));
            fprintf('模型配置\n');
            fprintf('%s\n', repmat('=', 1, 70));
            fprintf('\n型号: %s - %s\n', config.model_type, config.model_name);

            if isfield(config, 'description')
                fprintf('描述: %s\n', config.description);
            end

            % 架构参数
            fprintf('\n架构参数:\n');
            arch_fields = fieldnames(config.architecture);
            for i = 1:length(arch_fields)
                field = arch_fields{i};
                value = config.architecture.(field);
                if isnumeric(value)
                    if isscalar(value)
                        fprintf('  %s: %g\n', field, value);
                    else
                        fprintf('  %s: [%s]\n', field, num2str(value));
                    end
                else
                    fprintf('  %s: %s\n', field, string(value));
                end
            end

            % 训练参数
            fprintf('\n训练参数:\n');
            train_fields = fieldnames(config.training);
            for i = 1:length(train_fields)
                field = train_fields{i};
                if ~strcmp(field, 'early_stopping') && ~strcmp(field, 'scheduler')
                    value = config.training.(field);
                    if isnumeric(value)
                        fprintf('  %s: %g\n', field, value);
                    else
                        fprintf('  %s: %s\n', field, string(value));
                    end
                end
            end

            % 数据设置
            if isfield(config, 'data')
                fprintf('\n数据设置:\n');
                data_fields = fieldnames(config.data);
                for i = 1:length(data_fields)
                    field = data_fields{i};
                    value = config.data.(field);
                    if isnumeric(value)
                        fprintf('  %s: %g\n', field, value);
                    elseif islogical(value)
                        fprintf('  %s: %s\n', field, string(value));
                    else
                        fprintf('  %s: %s\n', field, string(value));
                    end
                end
            end

            % 特征筛选
            if isfield(config, 'feature_selection')
                fprintf('\n特征筛选: %s\n', string(config.feature_selection.enabled));
                if config.feature_selection.enabled
                    if isfield(config.feature_selection, 'correlation_threshold')
                        fprintf('  阈值: %g\n', config.feature_selection.correlation_threshold);
                    end
                    if isfield(config.feature_selection, 'top_k')
                        fprintf('  Top-K: %g\n', config.feature_selection.top_k);
                    end
                end
            end

            fprintf('%s\n', repmat('=', 1, 70));
        end
    end
end
