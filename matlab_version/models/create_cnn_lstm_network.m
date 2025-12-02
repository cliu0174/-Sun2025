function layers = create_cnn_lstm_network(input_size, cnn_channels, kernel_size, ...
    pool_size, lstm_hidden_size, lstm_num_layers, fc_hidden_sizes, dropout_rate)
% CREATE_CNN_LSTM_NETWORK 创建CNN-LSTM混合网络结构
%
% 输入参数:
%   input_size: 输入特征数量
%   cnn_channels: CNN通道数数组（例如 [32, 64]）
%   kernel_size: 卷积核大小
%   pool_size: 池化大小
%   lstm_hidden_size: LSTM隐藏层大小
%   lstm_num_layers: LSTM层数
%   fc_hidden_sizes: 全连接层大小数组
%   dropout_rate: Dropout比率
%
% 输出参数:
%   layers: LayerGraph对象，包含完整的CNN-LSTM网络

    % 默认参数
    if nargin < 1, input_size = 16; end
    if nargin < 2, cnn_channels = [32, 64]; end
    if nargin < 3, kernel_size = 3; end
    if nargin < 4, pool_size = 2; end
    if nargin < 5, lstm_hidden_size = 64; end
    if nargin < 6, lstm_num_layers = 2; end
    if nargin < 7, fc_hidden_sizes = [64]; end
    if nargin < 8, dropout_rate = 0.2; end

    padding_size = floor(kernel_size / 2);

    % 创建层数组
    layers = [
        sequenceInputLayer(input_size, 'Name', 'input')
    ];

    % 添加CNN层
    in_channels = input_size;
    for i = 1:length(cnn_channels)
        out_channels = cnn_channels(i);

        layers = [layers
            convolution1dLayer(kernel_size, out_channels, ...
                'Padding', padding_size, ...
                'Name', sprintf('conv%d', i))
            batchNormalizationLayer('Name', sprintf('bn%d', i))
            reluLayer('Name', sprintf('relu_conv%d', i))
            maxPooling1dLayer(pool_size, 'Stride', pool_size, ...
                'Name', sprintf('pool%d', i))
        ];

        in_channels = out_channels;
    end

    % 添加LSTM层
    for i = 1:lstm_num_layers
        if i == lstm_num_layers
            % 最后一层LSTM
            layers = [layers
                lstmLayer(lstm_hidden_size, ...
                    'OutputMode', 'last', ...
                    'Name', sprintf('lstm%d', i))
            ];
        else
            % 中间层LSTM
            layers = [layers
                lstmLayer(lstm_hidden_size, ...
                    'OutputMode', 'sequence', ...
                    'Name', sprintf('lstm%d', i))
            ];

            if lstm_num_layers > 1 && dropout_rate > 0
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
            reluLayer('Name', sprintf('relu_fc%d', i))
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
