function data = load_single_hust_battery(file_path, train_ratio, normalize_target, apply_cleaning)
% LOAD_SINGLE_HUST_BATTERY 加载单个HUST电池数据并划分训练/测试集
% 对应Python: data_loaders/data_loader_hust.py::load_single_hust_battery
%
% 输入:
%   file_path: CSV文件路径 (例如: 'data/HUST data/1-1.csv')
%   train_ratio: 训练集比例 (默认0.75)
%   normalize_target: 是否归一化目标值为SOH (默认true)
%   apply_cleaning: 是否应用3-Sigma清洗 (默认false)
%
% 输出:
%   data: 结构体
%       - train_features: (n_train, 16) 训练集特征
%       - train_capacity: (n_train, 1) 训练集容量
%       - test_features: (n_test, 16) 测试集特征
%       - test_capacity: (n_test, 1) 测试集容量
%       - scaler_mean: (1, 16) 标准化均值
%       - scaler_std: (1, 16) 标准化标准差
%       - battery_name: 电池名称
%       - feature_names: 特征名cell array
%       - n_train: 训练样本数
%       - n_test: 测试样本数
%       - rated_capacity: 额定容量 (1.1 Ah)
%       - normalize_target: 是否归一化
%       - cleaning_stats: 清洗统计（如果启用）

    % 默认参数
    if nargin < 2, train_ratio = 0.75; end
    if nargin < 3, normalize_target = true; end
    if nargin < 4, apply_cleaning = false; end

    % 读取CSV
    df = readtable(file_path);
    [~, battery_name, ~] = fileparts(file_path);

    % 应用3-Sigma清洗（可选）
    cleaning_stats = [];
    if apply_cleaning
        [df, cleaning_stats] = clean_3_sigma(df, false);
    end

    % 16个输入特征（对应Python的feature_columns）
    feature_columns = {
        'voltage_mean', 'voltage_std', 'voltage_kurtosis', 'voltage_skewness', ...
        'CC_Q', 'CC_charge_time', 'voltage_slope', 'voltage_entropy', ...
        'current_mean', 'current_std', 'current_kurtosis', 'current_skewness', ...
        'CV_Q', 'CV_charge_time', 'current_slope', 'current_entropy'
    };

    % 目标变量
    target_column = 'capacity';

    % 检查列是否存在
    df_columns = df.Properties.VariableNames;
    missing_cols = setdiff(feature_columns, df_columns);
    if ~isempty(missing_cols)
        error('Missing columns in %s: %s', file_path, strjoin(missing_cols, ', '));
    end

    if ~ismember(target_column, df_columns)
        error('Target column ''%s'' not found in %s', target_column, file_path);
    end

    % 按时序划分训练/测试集（对应Python逻辑）
    n_total = height(df);
    n_train = floor(n_total * train_ratio);

    train_df = df(1:n_train, :);
    test_df = df(n_train+1:end, :);

    % 提取特征和目标
    train_features = table2array(train_df(:, feature_columns));
    train_capacity = table2array(train_df(:, target_column));

    test_features = table2array(test_df(:, feature_columns));
    test_capacity = table2array(test_df(:, target_column));

    % 标准化特征（使用训练集的统计量，对应Python的StandardScaler）
    scaler_mean = mean(train_features, 1);
    scaler_std = std(train_features, 0, 1);

    % 防止除以零
    scaler_std(scaler_std == 0) = 1;

    train_features_scaled = (train_features - scaler_mean) ./ scaler_std;

    if ~isempty(test_features)
        test_features_scaled = (test_features - scaler_mean) ./ scaler_std;
    else
        test_features_scaled = zeros(0, length(feature_columns));
    end

    % 额定容量 (LFP电池: 1.1 Ah)
    rated_capacity = 1.1;

    % 归一化目标值为SOH（对应Python的normalize_target逻辑）
    if normalize_target
        % 方法1: 使用初始容量（更准确，推荐使用）- 对应Python注释
        initial_capacity = train_capacity(1);
        train_capacity_normalized = train_capacity / initial_capacity;
        if ~isempty(test_capacity)
            test_capacity_normalized = test_capacity / initial_capacity;
        else
            test_capacity_normalized = [];
        end
    else
        train_capacity_normalized = train_capacity;
        test_capacity_normalized = test_capacity;
    end

    % 构建输出结构体（对应Python的result字典）
    data = struct();
    data.train_features = train_features_scaled;
    data.train_capacity = train_capacity_normalized;
    data.test_features = test_features_scaled;
    data.test_capacity = test_capacity_normalized;
    data.scaler_mean = scaler_mean;
    data.scaler_std = scaler_std;
    data.battery_name = battery_name;
    data.feature_names = feature_columns;
    data.n_train = n_train;
    data.n_test = height(test_df);
    data.rated_capacity = rated_capacity;
    data.normalize_target = normalize_target;
    data.cleaning_stats = cleaning_stats;
end
