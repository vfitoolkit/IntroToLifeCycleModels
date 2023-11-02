function F=LifeCycleModel31_ReturnFn(savings,riskyshare,a,z,w,sigma,agej,Jr,pension,kappa_j)
% Note: riskyasset, so first inputs are (d,a,z,...)
% riskyshare is not used here, but still has to be input as it is part of d

F=-Inf;
if agej<Jr % If working age
    c=w*kappa_j*z+a-savings; % Add z here
else % Retirement
    c=pension+a-savings;
end

if c>0
    F=(c^(1-sigma))/(1-sigma); % The utility function
end

end
