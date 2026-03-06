classdef ModelFactory
    % MODELFACTORY 模型工厂类
    % 对应Python: models/model_factory.py::ModelFactory
    %
    % 提供统一接口创建所有支持的深度学习模型
    % 支持从配置文件加载或直接传入参数
    %
    % 用法:
    %   layers = ModelFactory.create_model('lstm', 16, 'configs/models/lstm_config.json');
    %   info = ModelFactory.get_model_info(layers);

    properties (Constant)
        % 支持的模型类型（对应Python的SUPPORTED_MODELS）
        SUPPORTED_MODELS = {'lstm', 'gru', 'cnn', 'cnn_lstm', 'bilstm', 'bigru'};
    end

    methods (Static)
        function layers = create_model(model_type, input_size, config_path, varargin)
            % 创建模型
            %
            % Args:
            %   model_type: 模型类型 ('lstm', 'gru', 'cnn', 'cnn_lstm')
            %   input_size: 输入特征维度
            %   config_path: 配置文件路径（可选）
            %   varargin: 额外参数（成对出现，用于覆盖配置）
            %
            % Returns:
            %   layers: Layer array

            model_type = lower(model_type);

            % 检查模型类型
            if ~ismember(model_type, ModelFactory.SUPPORTED_MODELS)
                error('Unsupported model type: %s. Supported: %s', ...
                    model_type, strjoin(ModelFactory.SUPPORTED_MODELS, ', '));
            end

            % 加载配置（对应Python的配置加载逻辑）
            if nargin < 3 || isempty(config_path)
                config = ConfigLoader.load_model_config(model_type);
            else
                config = ConfigLoader.load_config(config_path);
            end

            % 提取架构参数（对应Python: arch = config['architecture'].copy()）
            arch = config.architecture;
            arch.input_size = input_size;  % 覆盖为实际输入维度

            % 应用varargin覆盖（对应Python: arch.update(kwargs)）
            for i = 1:2:length(varargin)
                param_name = varargin{i};
                param_value = varargin{i+1};
                arch.(param_name) = param_value;
            end

            % 创建对应的模型（对应Python的if-elif链）
            switch model_type
                case 'lstm'
                    layers = ModelFactory.create_lstm(arch);
                case 'gru'
                    layers = ModelFactory.create_gru(arch);
                case 'cnn'
                    layers = ModelFactory.create_cnn(arch);
                case 'cnn_lstm'
                    layers = ModelFactory.create_cnn_lstm(arch);
                case 'bilstm'
                    layers = ModelFactory.create_bilstm(arch);
                case 'bigru'
                    layers = ModelFactory.create_bigru(arch);
                otherwise
                    error('Model type %s not implemented', model_type);
            end
        end

        function layers = create_lstm(arch)
            % 创建LSTM网络
            % 对应Python: models/baseline_models.py::LSTM

            input_size = arch.input_size;
            hidden_size = get_field_or_default(arch, 'hidden_size', 64);
            num_layers = get_field_or_default(arch, 'num_layers', 2);
            fc_hidden_sizes = get_field_or_default(arch, 'fc_hidden_sizes', [32, 16]);
            dropout_rate = get_field_or_default(arch, 'dropout_rate', 0.2);

            layers = [
                sequenceInputLayer(input_size, 'Name', 'input')
            ];

            % 添加LSTM层
            for i = 1:num_layers
                if i == num_layers
                    % 最后一层：输出last
                    layers = [layers
                        lstmLayer(hidden_size, 'OutputMode', 'last', ...
                            'Name', sprintf('lstm%d', i))
                    ];
                else
                    % 中间层：输出sequence
                    layers = [layers
                        lstmLayer(hidden_size, 'OutputMode', 'sequence', ...
                            'Name', sprintf('lstm%d', i))
                    ];
                    if num_layers > 1 && dropout_rate > 0
                        layers = [layers
                            dropoutLayer(dropout_rate, 'Name', sprintf('dropout_lstm%d', i))
                        ];
                    end
                end
            end

            % 全连接层
            for i = 1:length(fc_hidden_sizes)
                layers = [layers
                    fullyConnectedLayer(fc_hidden_sizes(i), 'Name', sprintf('fc%d', i))
                    reluLayer('Name', sprintf('relu%d', i))
                ];
                if dropout_rate > 0
                    layers = [layers
                        dropoutLayer(dropout_rate, 'Name', sprintf('dropout_fc%d', i))
                    ];
                end
            end

            % 输出层
            layers = [layers
                fullyConnectedLayer(1, 'Name', 'output')
                sigmoidLayer('Name', 'sigmoid')
                regressionLayer('Name', 'regression')
            ];
        end

        function layers = create_gru(arch)
            % 创建GRU网络
            % 对应Python: models/baseline_models.py::GRU

            input_size = arch.input_size;
            hidden_size = get_field_or_default(arch, 'hidden_size', 64);
            num_layers = get_field_or_default(arch, 'num_layers', 2);
            fc_hidden_sizes = get_field_or_default(arch, 'fc_hidden_sizes', [32, 16]);
            dropout_rate = get_field_or_default(arch, 'dropout_rate', 0.2);

            layers = [
                sequenceInputLayer(input_size, 'Name', 'input')
            ];

            % 添加GRU层
            for i = 1:num_layers
                if i == num_layers
                    layers = [layers
                        gruLayer(hidden_size, 'OutputMode', 'last', ...
                            'Name', sprintf('gru%d', i))
                    ];
                else
                    layers = [layers
                        gruLayer(hidden_size, 'OutputMode', 'sequence', ...
                            'Name', sprintf('gru%d', i))
                    ];
                    if num_layers > 1 && dropout_rate > 0
                        layers = [layers
                            dropoutLayer(dropout_rate, 'Name', sprintf('dropout_gru%d', i))
                        ];
                    end
                end
            end

            % 全连接层（与LSTM相同）
            for i = 1:length(fc_hidden_sizes)
                layers = [layers
                    fullyConnectedLayer(fc_hidden_sizes(i), 'Name', sprintf('fc%d', i))
                    reluLayer('Name', sprintf('relu%d', i))
                ];
                if dropout_rate > 0
                    layers = [layers
                        dropoutLayer(dropout_rate, 'Name', sprintf('dropout_fc%d', i))
                    ];
                end
            end

            layers = [layers
                fullyConnectedLayer(1, 'Name', 'output')
                sigmoidLayer('Name', 'sigmoid')
                regressionLayer('Name', 'regression')
            ];
        end

        function layers = create_cnn(arch)
            % 创建CNN网络
            % 对应Python: models/baseline_models.py::CNN

            input_size = arch.input_size;
            num_filters = get_field_or_default(arch, 'num_filters', 64);
            kernel_size = get_field_or_default(arch, 'kernel_size', 3);
            fc_hidden_sizes = get_field_or_default(arch, 'fc_hidden_sizes', [32, 16]);
            dropout_rate = get_field_or_default(arch, 'dropout_rate', 0.2);

            padding_size = floor(kernel_size / 2);

            layers = [
                sequenceInputLayer(input_size, 'Name', 'input')
                convolution1dLayer(kernel_size, num_filters, ...
                    'Padding', padding_size, 'Name', 'conv1')
                reluLayer('Name', 'relu_conv')
                globalMaxPooling1dLayer('Name', 'global_pool')
            ];

            % 全连接层
            for i = 1:length(fc_hidden_sizes)
                layers = [layers
                    fullyConnectedLayer(fc_hidden_sizes(i), 'Name', sprintf('fc%d', i))
                    reluLayer('Name', sprintf('relu%d', i))
                ];
                if dropout_rate > 0
                    layers = [layers
                        dropoutLayer(dropout_rate, 'Name', sprintf('dropout_fc%d', i))
                    ];
                end
            end

            layers = [layers
                fullyConnectedLayer(1, 'Name', 'output')
                sigmoidLayer('Name', 'sigmoid')
                regressionLayer('Name', 'regression')
            ];
        end

        function layers = create_cnn_lstm(arch)
            % 创建CNN-LSTM混合网络
            % 对应Python: models/cnn_lstm.py::CNN_LSTM

            input_size = arch.input_size;
            cnn_channels = get_field_or_default(arch, 'cnn_channels', [32, 64]);
            kernel_size = get_field_or_default(arch, 'kernel_size', 3);
            pool_size = get_field_or_default(arch, 'pool_size', 2);
            lstm_hidden_size = get_field_or_default(arch, 'hidden_size', 64);
            lstm_num_layers = get_field_or_default(arch, 'num_layers', 2);
            fc_hidden_sizes = get_field_or_default(arch, 'fc_hidden_sizes', [64]);
            dropout_rate = get_field_or_default(arch, 'dropout_rate', 0.2);

            padding_size = floor(kernel_size / 2);

            layers = [
                sequenceInputLayer(input_size, 'Name', 'input')
            ];

            % CNN部分
            for i = 1:length(cnn_channels)
                layers = [layers
                    convolution1dLayer(kernel_size, cnn_channels(i), ...
                        'Padding', padding_size, 'Name', sprintf('conv%d', i))
                    batchNormalizationLayer('Name', sprintf('bn%d', i))
                    reluLayer('Name', sprintf('relu_conv%d', i))
                    maxPooling1dLayer(pool_size, 'Stride', pool_size, ...
                        'Name', sprintf('pool%d', i))
                ];
            end

            % LSTM部分
            for i = 1:lstm_num_layers
                if i == lstm_num_layers
                    layers = [layers
                        lstmLayer(lstm_hidden_size, 'OutputMode', 'last', ...
                            'Name', sprintf('lstm%d', i))
                    ];
                else
                    layers = [layers
                        lstmLayer(lstm_hidden_size, 'OutputMode', 'sequence', ...
                            'Name', sprintf('lstm%d', i))
                    ];
                    if lstm_num_layers > 1 && dropout_rate > 0
                        layers = [layers
                            dropoutLayer(dropout_rate, 'Name', sprintf('dropout_lstm%d', i))
                        ];
                    end
                end
            end

            % 全连接层
            for i = 1:length(fc_hidden_sizes)
                layers = [layers
                    fullyConnectedLayer(fc_hidden_sizes(i), 'Name', sprintf('fc%d', i))
                    reluLayer('Name', sprintf('relu_fc%d', i))
                ];
                if dropout_rate > 0
                    layers = [layers
                        dropoutLayer(dropout_rate, 'Name', sprintf('dropout_fc%d', i))
                    ];
                end
            end

            layers = [layers
                fullyConnectedLayer(1, 'Name', 'output')
                sigmoidLayer('Name', 'sigmoid')
                regressionLayer('Name', 'regression')
            ];
        end

        function layers = create_bilstm(arch)
            % 创建BiLSTM网络（双向LSTM）
            % 对应Python: models/baseline_models.py::BiLSTM

            input_size = arch.input_size;
            hidden_size = get_field_or_default(arch, 'hidden_size', 64);
            num_layers = get_field_or_default(arch, 'num_layers', 2);
            fc_hidden_sizes = get_field_or_default(arch, 'fc_hidden_sizes', [32, 16]);
            dropout_rate = get_field_or_default(arch, 'dropout_rate', 0.2);

            layers = [
                sequenceInputLayer(input_size, 'Name', 'input')
            ];

            % 添加双向LSTM层
            for i = 1:num_layers
                if i == num_layers
                    layers = [layers
                        bilstmLayer(hidden_size, 'OutputMode', 'last', ...
                            'Name', sprintf('bilstm%d', i))
                    ];
                else
                    layers = [layers
                        bilstmLayer(hidden_size, 'OutputMode', 'sequence', ...
                            'Name', sprintf('bilstm%d', i))
                    ];
                    if num_layers > 1 && dropout_rate > 0
                        layers = [layers
                            dropoutLayer(dropout_rate, 'Name', sprintf('dropout_bilstm%d', i))
                        ];
                    end
                end
            end

            % 全连接层（注意：BiLSTM输出维度是hidden_size*2）
            for i = 1:length(fc_hidden_sizes)
                layers = [layers
                    fullyConnectedLayer(fc_hidden_sizes(i), 'Name', sprintf('fc%d', i))
                    reluLayer('Name', sprintf('relu%d', i))
                ];
                if dropout_rate > 0
                    layers = [layers
                        dropoutLayer(dropout_rate, 'Name', sprintf('dropout_fc%d', i))
                    ];
                end
            end

            layers = [layers
                fullyConnectedLayer(1, 'Name', 'output')
                sigmoidLayer('Name', 'sigmoid')
                regressionLayer('Name', 'regression')
            ];
        end

        function layers = create_bigru(arch)
            % 创建BiGRU网络（双向GRU）
            % 对应Python: models/baseline_models.py::BiGRU

            input_size = arch.input_size;
            hidden_size = get_field_or_default(arch, 'hidden_size', 64);
            num_layers = get_field_or_default(arch, 'num_layers', 2);
            fc_hidden_sizes = get_field_or_default(arch, 'fc_hidden_sizes', [32, 16]);
            dropout_rate = get_field_or_default(arch, 'dropout_rate', 0.2);

            layers = [
                sequenceInputLayer(input_size, 'Name', 'input')
            ];

            % 添加双向GRU层
            for i = 1:num_layers
                if i == num_layers
                    layers = [layers
                        bigruLayer(hidden_size, 'OutputMode', 'last', ...
                            'Name', sprintf('bigru%d', i))
                    ];
                else
                    layers = [layers
                        bigruLayer(hidden_size, 'OutputMode', 'sequence', ...
                            'Name', sprintf('bigru%d', i))
                    ];
                    if num_layers > 1 && dropout_rate > 0
                        layers = [layers
                            dropoutLayer(dropout_rate, 'Name', sprintf('dropout_bigru%d', i))
                        ];
                    end
                end
            end

            % 全连接层
            for i = 1:length(fc_hidden_sizes)
                layers = [layers
                    fullyConnectedLayer(fc_hidden_sizes(i), 'Name', sprintf('fc%d', i))
                    reluLayer('Name', sprintf('relu%d', i))
                ];
                if dropout_rate > 0
                    layers = [layers
                        dropoutLayer(dropout_rate, 'Name', sprintf('dropout_fc%d', i))
                    ];
                end
            end

            layers = [layers
                fullyConnectedLayer(1, 'Name', 'output')
                sigmoidLayer('Name', 'sigmoid')
                regressionLayer('Name', 'regression')
            ];
        end

        function count = count_parameters(layers)
            % 计算模型参数数量
            % 对应Python: sum(p.numel() for p in model.parameters())

            % 注意：MATLAB的Layer array没有直接获取参数的方法
            % 这里返回层数作为简化
            count = length(layers);
        end

        function info = get_model_info(layers)
            % 获取模型信息
            % 对应Python: ModelFactory.get_model_info

            info = struct();
            info.num_layers = length(layers);
            info.layer_types = cell(length(layers), 1);
            for i = 1:length(layers)
                info.layer_types{i} = class(layers(i));
            end
        end
    end
end

% 辅助函数
function value = get_field_or_default(struct_var, field_name, default_value)
    % 获取结构体字段，如果不存在则返回默认值
    % 对应Python: arch.get('field_name', default_value)

    if isfield(struct_var, field_name)
        value = struct_var.(field_name);
    else
        value = default_value;
    end
end
