function y = demean(x)
    [m,n] = size(x);
    if m < n
        x = x';
        y = detrend(x,'constant');
        y = y';
    else
        y = detrend(x,'constant');
    end
    
    
    
    