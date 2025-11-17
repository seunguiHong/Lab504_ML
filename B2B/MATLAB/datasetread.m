%% ==== (0) SETTINGS (minimal) ============================================
DATA_FILE    = 'dataset.csv';         % CSV 경로(또는 T를 미리 제공)
PERIOD_START = 197108;                % YYYYMM (inclusive)
PERIOD_END   = 202312;                % YYYYMM (inclusive)
MATURITIES   = ["xr_2","xr_3","xr_5","xr_7","xr_10"];  % 타깃 열(순서 유지)
BURN_END     = 199001;                % burn-in end (YYYYMM)
H            = 12;                    % forecast horizon (months)
% 선택: 슬로프 외에 반드시 포함할 X 열 이름들(순서 유지). 없으면 자동 선택.
% X_INCLUDE = ["fwd_2","ip"];   % 필요 시 사용

%% ==== (1) READ & CLEAN ===================================================
if ~(exist('T','var') && istable(T))
    o = detectImportOptions(DATA_FILE, 'TextType','string'); try, o.VariableNamingRule='preserve'; catch, end
    T = readtable(DATA_FILE, o);
end
tc = T.Time;
if isstring(tc) || iscellstr(tc)
    s = string(tc); yy = str2double(extractBefore(s,5)); mm = str2double(extractAfter(s,4)); T.Time = double(yy*100+mm);
else
    T.Time = double(tc);
end
T = sortrows(T,'Time');
T = T(T.Time>=PERIOD_START & T.Time<=PERIOD_END, :);
assert(~isempty(T), 'No rows in the requested period.');

%% ==== (2) BUILD Y and X (slopes first; no separate S) ====================
Y_cols = string(MATURITIES);  J = numel(Y_cols);
assert(all(ismember(Y_cols, string(T.Properties.VariableNames))), 'Missing Y columns in T.');

% slope 이름 감지: xr_k -> {s_k, s+k, slope_k, ...} 중 존재하는 첫 후보
VN = string(T.Properties.VariableNames);
S_cols = strings(J,1);
for j = 1:J
    k = regexp(Y_cols(j), '\d+', 'match','once');
    cand = ["s_"+k,"s"+k,"slope_"+k,"slope"+k,"S_"+k,"S"+k];
    hit  = cand(ismember(cand, VN));
    assert(~isempty(hit), "Slope column for %s not found.", Y_cols(j));
    S_cols(j) = hit(1);
end

% 기타 X 피처 선택: 사용자가 X_INCLUDE 지정 시 그 순서대로, 아니면 자동(숫자/논리형 전부)
exclude = unique([{'Time'}; cellstr(Y_cols(:)); cellstr(S_cols(:))], 'stable');
if exist('X_INCLUDE','var') && ~isempty(X_INCLUDE)
    inc = string(X_INCLUDE(:));
    inc = inc(~ismember(inc, ["Time"; Y_cols(:); S_cols(:)]));
    missing = inc(~ismember(inc, VN));  assert(isempty(missing), 'X_INCLUDE not found: %s', strjoin(cellstr(missing), ', '));
    inc_cell = cellstr(inc);
    isnum = cellfun(@(nm) isnumeric(T.(nm)) || islogical(T.(nm)), inc_cell);
    assert(all(isnum), 'X_INCLUDE must be numeric/logical.');
    vkeep = inc_cell(:);
else
    keep  = setdiff(T.Properties.VariableNames, exclude, 'stable');
    vkeep = keep( cellfun(@(v) isnumeric(T.(v)) || islogical(T.(v)), keep) );
    vkeep = vkeep(:);
end

% X/Y 테이블 → 행렬 (X는 [slopes | other] 순서, slopes가 항상 선두 J열)
vars_slopes = [{'Time'}; cellstr(S_cols(:))];
vars_y      = [{'Time'}; cellstr(Y_cols(:))];
Xtbl = T(:, [vars_slopes; vkeep]);          % [Time | s_* ... | other ...]
Ytbl = T(:,  vars_y);
Time = Xtbl.Time;
Y    = double(table2array(Ytbl(:,2:end)));  % TN x J
X    = double(table2array(Xtbl(:,2:end)));  % TN x p
X_VAR_NAMES = string(Xtbl.Properties.VariableNames(2:end));
[TN, p] = size(X);

% slopes-first 검증
lhs = string(X_VAR_NAMES(1:J)); lhs = lhs(:)';  rhs = string(S_cols(:))'; 
assert(all(lhs == rhs), 'Slopes must be the first J columns of X.');

% burn-in/OOS
idx_burn_end = find(Time >= BURN_END, 1, 'first');  assert(~isempty(idx_burn_end), 'BURN_END not in sample.');
nOOS = TN - idx_burn_end - H;  assert(nOOS>0, 'Not enough OOS after burn-in & H.');
fprintf('[DATA] %d–%d | T=%d | J=%d | p=%d | H=%d | burn_end@%d | nOOS=%d\n', ...
    min(Time), max(Time), TN, J, p, H, BURN_END, nOOS);

%% ==== (3) H-step alignment (ALL maturities at once) ======================
% X_t → Y_{t+H}
TimeH = Time(1+H:end);     % (TN-H) x 1
XH    = X(1:end-H,   :);   % (TN-H) x p   (첫 J열 = slopes)
YH    = Y(1+H:end,   :);   % (TN-H) x J

idx_burn_end_H = find(TimeH >= BURN_END, 1, 'first');
train_ix = 1:idx_burn_end_H;
oos_ix   = (idx_burn_end_H+1):numel(TimeH);
fprintf('[ALIGN] TN_H=%d | train=%d | OOS=%d\n', numel(TimeH), numel(train_ix), numel(oos_ix));