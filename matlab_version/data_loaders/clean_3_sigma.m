function [df_cleaned, stats] = clean_3_sigma(df, verbose)
% CLEAN_3_SIGMA 使用3-Sigma规则清洗异常值
% 对应Python: data_loaders/data_loader_hust.py::clean_3_sigma
%
% 规则: 删除超过 mean ± 3*std 的数据点
%
% 输入:
%   df: MATLAB table
%   verbose: 是否打印信息 (默认false)
%
% 输出:
%   df_cleaned: 清洗后的table
%   stats: 统计信息结构体

    if nargin < 2
        verbose = false;
    end

    original_size = height(df);

    % 1. 替换Inf为NaN（对应Python: df.replace([np.inf, -np.inf], np.nan)）
    df_cleaned = df;
    for col_idx = 1:width(df_cleaned)
        col_data = table2array(df_cleaned(:, col_idx));
        if isnumeric(col_data)
            col_data(isinf(col_data)) = NaN;
            df_cleaned{:, col_idx} = col_data;
        end
    end

    % 2. 删除含NaN的行（对应Python: df.dropna()）
    df_cleaned = rmmissing(df_cleaned);
    after_nan = height(df_cleaned);

    % 3. 对每列应用3-Sigma规则（对应Python的for col in df.columns）
    removed_by_column = containers.Map('KeyType', 'char', 'ValueType', 'double');

    for col_idx = 1:width(df_cleaned)
        col_name = df_cleaned.Properties.VariableNames{col_idx};
        col_data = table2array(df_cleaned(:, col_idx));

        if ~isnumeric(col_data)
            continue;
        end

        before = height(df_cleaned);

        % 计算均值和标准差
        col_mean = mean(col_data);
        col_std = std(col_data);

        % 计算上下界（对应Python: mean ± 3 * std）
        lower_bound = col_mean - 3 * col_std;
        upper_bound = col_mean + 3 * col_std;

        % 删除超过界限的行
        valid_rows = (col_data >= lower_bound) & (col_data <= upper_bound);
        df_cleaned = df_cleaned(valid_rows, :);

        removed = before - height(df_cleaned);
        if removed > 0
            removed_by_column(col_name) = removed;
        end
    end

    final_size = height(df_cleaned);

    % 统计信息（对应Python的stats字典）
    stats = struct();
    stats.original_size = original_size;
    stats.after_nan_removal = after_nan;
    stats.final_size = final_size;
    stats.total_removed = original_size - final_size;
    stats.nan_removed = original_size - after_nan;
    stats.outliers_removed = after_nan - final_size;

    if original_size > 0
        stats.removal_rate = (original_size - final_size) / original_size * 100;
    else
        stats.removal_rate = 0;
    end

    stats.removed_by_column = removed_by_column;

    % 打印信息（对应Python的verbose逻辑）
    if verbose
        fprintf('  3-Sigma清洗: %d -> %d (删除 %d, %.1f%%)\n', ...
            original_size, final_size, stats.total_removed, stats.removal_rate);
    end
end
