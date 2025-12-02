function layers = create_cnn_network(input_size, num_filters, kernel_size, fc_hidden_sizes, dropout_rate)
% CREATE_CNN_NETWORK 创建1D CNN网络结构
%
% 输入参数:
%   input_size: 输入特征数量
%   num_filters: 卷积核数量
%   kernel_size: 卷积核大小
%   fc_hidden_sizes: 全连接层大小数组（例如 [32, 16]）
%   dropout_rate: Dropout比率
%
% 输出参数:
%   layers: LayerGraph对象，包含完整的CNN网络

    % 默认参数
    if nargin < 1, input_size = 16; end
    if nargin < 2, num_filters = 64; end
    if nargin < 3, kernel_size = 3; end
    if nargin < 4, fc_hidden_sizes = [32, 16]; end
    if nargin < 5, dropout_rate = 0.2; end

    % 计算padding以保持输出尺寸
    padding_size = floor(kernel_size / 2);

    % 创建层数组
    layers = [
        sequenceInputLayer(input_size, 'Name', 'input')

        % 1D卷积层
        convolution1dLayer(kernel_size, num_filters, ...
            'Padding', padding_size, ...
            'Name', 'conv1')
        reluLayer('Name', 'relu_conv')

        % 全局最大池化
        globalMaxPooling1dLayer('Name', 'global_pool')
    ];

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
