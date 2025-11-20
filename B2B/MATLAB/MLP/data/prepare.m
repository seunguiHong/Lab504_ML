function out = prepare(T_or_path, varargin)
% prepare: 데이터셋 → (동일 시점) Y, X 생성. X는 슬로프 선두 고정.
% Usage:
%   out = prepare('dataset.csv', ...
%       'Period',[197108 202312], ...
%       'Maturities',["xr_2","xr_3","xr_5","xr_7","xr_10"], ...
%       'BurnEnd',199001, ...
%       'XInclude',["fwd_2","ip"]);     % 선택: 추가로 포함할 X 열 이름들
%
% Notes:
% - 항상 같은 시점의 X_t, Y_t를 사용한다. (H-step 정렬 없음)
% - Y가 이미 미래 목표로 설계되어 있다고 가정한다. (H는 받더라도 미사용)

% ---- defaults / parse name-value ----------------------------------------
P = struct('Period',[197108 202312], ...
           'Maturities',["xr_2","xr_3","xr_5"], ...
           'BurnEnd',199001, ...
           'H',[] , ...             % 호환성만 유지, 미사용
           'XInclude',[]);
for k = 1:2:numel(varargin)
    P.(varargin{k}) = varargin{k+1};
end
PERIOD_START = P.Period(1); PERIOD_END = P.Period(2);
Y_cols = string(P.Maturities); J = numel(Y_cols);

% ---- acquire table T -----------------------------------------------------
if istable(T_or_path)
    T = T_or_path;
else
    o = detectImportOptions(T_or_path, 'TextType','string'); try, o.VariableNamingRule='preserve'; catch, end
    T = readtable(T_or_path, o);
end
assert(any(strcmp('Time', T.Properties.VariableNames)), 'T must contain Time.');

% ---- normalize Time(YYYYMM) & filter ------------------------------------
tc = T.Time;
if isstring(tc) || iscellstr(tc)
    s = string(tc);
    ok = ~cellfun('isempty', regexp(cellstr(s), '^\d{6}$', 'once'));
    assert(all(ok), 'Time must be 6-digit YYYYMM.');
    yy = str2double(extractBefore(s,5));
    mm = str2double(extractAfter(s,4));  assert(all(mm>=1 & mm<=12));
    T.Time = double(yy*100 + mm);
else
    T.Time = double(tc);
end
T = sortrows(T, 'Time');
T = T(T.Time>=PERIOD_START & T.Time<=PERIOD_END, :);
assert(~isempty(T), 'No rows in the requested period.');
VN = string(T.Properties.VariableNames);

% ---- check targets & detect slope names ---------------------------------
assert(all(ismember(Y_cols, VN)), 'Missing Y columns in T.');
S_cols = strings(J,1);
for j = 1:J
    k = regexp(Y_cols(j), '\d+', 'match','once');                 % maturity number
    cand = ["s_"+k,"s"+k,"slope_"+k,"slope"+k,"S_"+k,"S"+k];
    hit  = cand(ismember(cand, VN));
    assert(~isempty(hit), "Slope column for %s not found.", Y_cols(j));
    S_cols(j) = hit(1);
end

% ---- choose other X features --------------------------------------------
exclude = unique([{'Time'}; cellstr(Y_cols(:)); cellstr(S_cols(:))], 'stable');
if ~isempty(P.XInclude)
    inc = string(P.XInclude(:));
    inc = inc(~ismember(inc, ["Time"; Y_cols(:); S_cols(:)]));   % 금지/중복 제거
    miss = inc(~ismember(inc, VN));
    assert(isempty(miss), 'XInclude not found: %s', strjoin(cellstr(miss), ', '));
    inc_cell = cellstr(inc);
    oknum = cellfun(@(nm) isnumeric(T.(nm)) || islogical(T.(nm)), inc_cell);
    assert(all(oknum), 'XInclude must be numeric/logical.');
    vkeep = inc_cell(:);                                         % 지정 순서 유지
else
    keep  = setdiff(T.Properties.VariableNames, exclude, 'stable');
    vkeep = keep( cellfun(@(v) isnumeric(T.(v)) || islogical(T.(v)), keep) );
    vkeep = vkeep(:);
end

% ---- build X/Y tables and convert (slopes first) -------------------------
vars_slopes = [{'Time'}; cellstr(S_cols(:))];
vars_y      = [{'Time'}; cellstr(Y_cols(:))];

Xtbl = T(:, [vars_slopes; vkeep]);              % [Time | slopes... | other...]
Ytbl = T(:,  vars_y);

Time = Xtbl.Time;
Y    = double(table2array(Ytbl(:,2:end)));      % TN x J
X    = double(table2array(Xtbl(:,2:end)));      % TN x p  (first J = slopes)
X_VAR_NAMES = string(Xtbl.Properties.VariableNames(2:end));
[TN, p] = size(X);

% ---- slopes-first sanity -------------------------------------------------
lhs = string(X_VAR_NAMES(1:J)); lhs = lhs(:)';  rhs = string(S_cols(:))';
assert(all(lhs == rhs), 'Slopes must be the first J columns of X.');

% ---- burn-in / OOS (same-time pairing) ----------------------------------
idx_burn_end = find(Time >= P.BurnEnd, 1, 'first');
assert(~isempty(idx_burn_end), 'BurnEnd not in sample.');
nOOS = TN - idx_burn_end;                    

% ---- pack outputs --------------------------------------------------------
out = struct('Time',Time,'Y',Y,'X',X,'X_VAR_NAMES',X_VAR_NAMES, ...
             'TN',TN,'J',J,'p',p, ...
             'idx_burn_end',idx_burn_end,'nOOS',nOOS, ...
             'S_cols',S_cols);
end