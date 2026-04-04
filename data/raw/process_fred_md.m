input_file  = 'raw/2025-12-MD.csv';
output_file = 'fred_md_processed.csv';

C = readcell(input_file, "Delimiter", ",");

series = string(C(1, 2:end));
tcode  = str2double(string(C(2, 2:end)));

dcol = C(3:end, 1);

if isa(dcol{1}, "datetime")
    dt = vertcat(dcol{:});
    Time = string(dt, "yyyyMM");
elseif isnumeric(dcol{1})
    dt = datetime(cell2mat(dcol), "ConvertFrom", "excel");
    Time = string(dt, "yyyyMM");
else
    dstr = strip(string(dcol));
    s0 = dstr(find(strlength(dstr) > 0, 1, "first"));

    if contains(s0, "/")
        parts = split(dstr, "/");
        mm = compose("%02d", str2double(parts(:, 1)));
        yy_raw = strip(parts(:, 3));
        yy_num = str2double(yy_raw);
        yy = compose("%04d", yy_num + (yy_num < 100) * 2000);
        Time = yy + mm;
    else
        if contains(s0, "-") && any(isstrprop(s0, "alpha"))
            dt = datetime(dstr, "InputFormat", "dd-MMM-uuuu");
        elseif contains(s0, "-")
            dt = datetime(dstr, "InputFormat", "uuuu-MM-dd");
        else
            dt = datetime(dstr);
        end
        Time = string(dt, "yyyyMM");
    end
end

X = str2double(string(C(3:end, 2:end)));
Y = apply_tcode_matrix(X, tcode);

T = array2table(Y, "VariableNames", cellstr(series));
T.Time = Time;
T = movevars(T, "Time", "Before", 1);

writetable(T, output_file);

function Y = apply_tcode_matrix(X, tcode)
n = size(X, 1);
p = size(X, 2);
Y = NaN(n, p);
for j = 1:p
    Y(:, j) = apply_tcode_vector(X(:, j), tcode(j));
end
end

function y = apply_tcode_vector(x, code)
n = size(x, 1);
y = NaN(n, 1);
small = 1e-6;

switch code
    case 1
        y = x;
    case 2
        y(2:n) = x(2:n) - x(1:n-1);
    case 3
        y(3:n) = x(3:n) - 2*x(2:n-1) + x(1:n-2);
    case 4
        if min(x) < small
            y = NaN;
        else
            y = log(x);
        end
    case 5
        if min(x) > small
            lx = log(x);
            y(2:n) = lx(2:n) - lx(1:n-1);
        end
    case 6
        if min(x) > small
            lx = log(x);
            y(3:n) = lx(3:n) - 2*lx(2:n-1) + lx(1:n-2);
        end
    case 7
        y1 = NaN(n, 1);
        y1(2:n) = (x(2:n) - x(1:n-1)) ./ x(1:n-1);
        y(3:n)  = y1(3:n) - y1(2:n-1);
end
end