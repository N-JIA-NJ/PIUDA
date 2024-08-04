function y=sig(x)
    y=1./(1.0+exp(-10*(x-0.5)));
    y=1-y;
end