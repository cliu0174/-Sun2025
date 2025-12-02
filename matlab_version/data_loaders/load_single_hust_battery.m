function data = load_single_hust_battery(file_path, train_ratio, normalize_target, apply_cleaning)
% LOAD_SINGLE_HUST_BATTERY 加载单个HUST电池数据并划分训练/测试集
%
% 输入参数:
%   file_path: CSV文件路径 (例如: 'data/HUST data/1-1.csv')
%   train_ratio: 训练集比例 (默认0.75，即75%)
%   normalize_target: 是否归一化目标值为SOH (默认true)
%   apply_cleaning: 是否应用3-Sigma清洗 (默认false)
%
% 输出参数:
%   data: 结构体包含:
%       - train_features: 训练集特征 (归一化后)
%       - train_capacity: 训练集容量 (归一化为SOH)
%       - test_features: 测试集特征 (归一化后)
%       - test_capacity: 测试集容量 (归一化为SOH)
%       - scaler_mean: 标准化均值
%       - scaler_std: 标准化标准差
%       - battery_name: 电池名称
%       - feature_names: 特征名列表
%       - rated_capacity: 额定容量 (1.1 Ah)
%       - cleaning_stats: 清洗统计信息

    % 设置默认参数
    if nargin < 2, train_ratio = 0.75; end
    if nargin < 3, normalize_target = true; end
    if nargin < 4, apply_cleaning = false; end

    % 读取CSV文件
    df = readtable(file_path);
    [~, name, ~] = fileparts(file_path);
    battery_name = name;

    % 应用3-Sigma清洗（可选）
    cleaning_stats = [];
    if apply_cleaning
        [df, cleaning_stats] = clean_3_sigma(df, false);
    end

    % 16个输入特征
    feature_columns = {
        'voltage_mean', 'voltage_std', 'voltage_kurtosis', 'voltage_skewness', ...
        'CC_Q', 'CC_charge_time', 'voltage_slope', 'voltage_entropy', ...
        'current_mean', 'current_std', 'current_kurtosis', 'current_skewness', ...
        'CV_Q', 'CV_charge_time', 'current_slope', 'current_entropy'
    };

    % 目标变量
    target_column = 'capacity';

    % 检查列是否存在（MATLAB列名自动转换空格为下划线）
    df_columns = df.Properties.VariableNames;
    missing_cols = setdiff(feature_columns, df_columns);
    if ~isempty(missing_cols)
        error('Missing columns in %s: %s', file_path, strjoin(missing_cols, ', '));
    end

    if ~ismember(target_column, df_columns)
        error('Target column ''%s'' not found in %s', target_column, file_path);
    end

    % 按时序划分训练/测试集
    n_total = height(df);
    n_train = floor(n_total * train_ratio);

    train_df = df(1:n_train, :);
    test_df = df(n_train+1:end, :);

    % 提取特征和目标
    train_features = table2array(train_df(:, feature_columns));
    train_capacity = table2array(train_df(:, target_column));

    test_features = table2array(test_df(:, feature_columns));
    test_capacity = table2array(test_df(:, target_column));

    % 标准化特征（使用训练集的统计量）
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

    % 归一化目标值为SOH
    if normalize_target
        % 使用初始容量归一化（推荐方法）
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

    % 构建输出结构体
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
