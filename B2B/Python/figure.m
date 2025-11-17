%% main_hov.m
% *_mse10.csv 파일을 자동으로 찾아서
% - Ridge vs CSARM(Ridge)
% - Plain MLP vs SDMLP
% 를 fwd / fwd+macro 각각 비교하고 Figure 생성

clear; clc;

% 이 스크립트가 있는 폴더를 루트로 사용
rootDir = fileparts(mfilename('fullpath'));
fprintf('Root directory: %s\n\n', rootDir);

% label, file1, file2 (파일명만; 위치는 자동 검색)
pairs = {
    'fwd: Ridge vs CSARM(Ridge)',       'ridge_fwd_mse10.csv',       'csarm_ridge_fwd_mse10.csv';
    'fwd: Plain MLP vs SDMLP',          'mlp_fwd_mse10.csv',         'sdmlp_fwd_mse10.csv';
    'fwd+macro: Ridge vs CSARM(Ridge)', 'ridge_fwd_macro_mse10.csv', 'csarm_ridge_fwd_macro_mse10.csv';
    'fwd+macro: Plain MLP vs SDMLP',    'mlp_fwd_macro_mse10.csv',   'sdmlp_fwd_macro_mse10.csv';
};

nPairs = size(pairs, 1);
fprintf('Number of pairs: %d\n\n', nPairs);

for i = 1:nPairs
    label  = pairs{i, 1};
    fname1 = pairs{i, 2};
    fname2 = pairs{i, 3};

    % --- 파일 위치 자동 탐색 ---
    file1 = find_file_recursive(rootDir, fname1);
    file2 = find_file_recursive(rootDir, fname2);

    if isempty(file1)
        fprintf('*** Skip "%s": file not found: %s ***\n\n', label, fname1);
        continue;
    end
    if isempty(file2)
        fprintf('*** Skip "%s": file not found: %s ***\n\n', label, fname2);
        continue;
    end

    s1 = read_mse10_csv(file1);
    s2 = read_mse10_csv(file2);

    % 공통 구간으로 정렬 (datetime 기준)
    [tCommon, idx1, idx2] = intersect(s1.time, s2.time, 'stable');
    mse1 = s1.mse(idx1);
    mse2 = s2.mse(idx2);

    mean1 = mean(mse1);
    mean2 = mean(mse2);
    diffMean   = mean2 - mean1;        % >0이면 model2가 더 나쁨
    relImprove = 1 - mean2 / mean1;    % >0이면 model2가 더 좋음

    fprintf('=== %s ===\n', label);
    fprintf('Model 1: %s\n', fname1);
    fprintf('Model 2: %s\n', fname2);
    fprintf('Mean MSE_10 (model 1): %.6f\n', mean1);
    fprintf('Mean MSE_10 (model 2): %.6f\n', mean2);
    fprintf('Diff (model2 - model1): %.6f\n', diffMean);
    fprintf('Relative improvement (1 - M2/M1): %.4f\n\n', relImprove);

    % Figure 생성 (x축: datetime)
    figure('Name', ['MSE_10 - ' label], 'NumberTitle', 'off');
    hold on;
    plot(tCommon, mse1, '-',  'LineWidth', 1.5);
    plot(tCommon, mse2, '--', 'LineWidth', 1.5);
    hold off;
    grid on;
    xlabel('Time');
    ylabel('MSE_{10}');
    legend({'Model 1','Model 2'}, 'Location','best');
    title(['MSE_{10} comparison: ' label], 'Interpreter','none');

    % tick 포맷 (연도 혹은 연-월)
    ax = gca;
    % 년도만 보고 싶으면:
    % ax.XAxis.TickLabelFormat = 'yyyy';
    % 연-월까지 보고 싶으면:
    ax.XAxis.TickLabelFormat = 'yyyy-MM';
end

%% local function: 재귀적으로 파일 찾기
function fullpath = find_file_recursive(rootDir, filename)
    % rootDir 아래에서 filename을 재귀적으로 탐색
    d = dir(fullfile(rootDir, '**', filename));  % R2016b+ 에서 지원
    if isempty(d)
        fullpath = '';
    else
        fullpath = fullfile(d(1).folder, d(1).name); % 첫 번째 매치 사용
    end
end

%% local function: CSV 읽기 (YYYYMM -> datetime 변환)
function s = read_mse10_csv(filePath)
    % CSV 포맷:
    %   첫 번째 컬럼: Time (YYYYMM, 문자열/숫자)
    %   두 번째 컬럼: MSE_10

    if ~exist(filePath, 'file')
        error('File not found: %s', filePath);
    end

    T = readtable(filePath, 'Delimiter', ',', 'ReadVariableNames', true);

    % 첫 번째 컬럼: 시간 (string 또는 numeric)
    timeCol = T{:,1};
    if iscellstr(timeCol) || isstring(timeCol)
        timeNum = str2double(string(timeCol));
    else
        timeNum = timeCol;
    end
    timeNum = timeNum(:);

    % YYYYMM -> year, month -> datetime(년, 월, 1일)
    year  = floor(timeNum ./ 100);
    month = timeNum - year * 100;
    % day = 1 로 고정 (월 데이터이므로)
    timeDT = datetime(year, month, 1);

    % 두 번째 컬럼: MSE_10 (이름이 MSE_10이라고 가정)
    if ismember('MSE_10', T.Properties.VariableNames)
        mse = T.MSE_10;
    else
        mse = T{:,2};
    end

    s.time = timeDT(:);   % datetime 벡터
    s.mse  = mse(:);
end