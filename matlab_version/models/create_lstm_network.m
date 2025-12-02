function layers = create_lstm_network(input_size, hidden_size, num_layers, fc_hidden_sizes, dropout_rate)
% CREATE_LSTM_NETWORK 创建LSTM网络结构
%
% 输入参数:
%   input_size: 输入特征数量（每个时间步）
%   hidden_size: LSTM隐藏层大小
%   num_layers: LSTM层数
%   fc_hidden_sizes: 全连接层大小数组（例如 [32, 16]）
%   dropout_rate: Dropout比率
%
% 输出参数:
%   layers: LayerGraph对象，包含完整的LSTM网络

    % 默认参数
    if nargin < 1, input_size = 16; end
    if nargin < 2, hidden_size = 64; end
    if nargin < 3, num_layers = 2; end
    if nargin < 4, fc_hidden_sizes = [32, 16]; end
    if nargin < 5, dropout_rate = 0.2; end

    % 创建层数组
    layers = [
        sequenceInputLayer(input_size, 'Name', 'input')
    ];

    % 添加LSTM层
    for i = 1:num_layers
        layer_name = sprintf('lstm%d', i);

        if i == num_layers
            % 最后一层LSTM只输出最后的隐藏状态
            layers = [layers
                lstmLayer(hidden_size, ...
                    'OutputMode', 'last', ...
                    'Name', layer_name)
            ];
        else
            % 中间层LSTM输出完整序列
            layers = [layers
                lstmLayer(hidden_size, ...
                    'OutputMode', 'sequence', ...
                    'Name', layer_name)
            ];

            % 在中间层之间添加Dropout
            if num_layers > 1 && dropout_rate > 0
                layers = [layers
                    dropoutLayer(dropout_rate, 'Name', sprintf('dropout_lstm%d', i))
                ];
            end
        end
    end

    % 添加全连接层
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
        sigmoidLayer('Name', 'sigmoid')  % SOH在[0,1]范围内
        regressionLayer('Name', 'regression')
    ];
end
