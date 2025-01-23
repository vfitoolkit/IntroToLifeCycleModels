function F=LifeCycleModel31_ReturnFn(savings,a,z,w,sigma,agej,Jr,pension,kappa_j)
% Note: riskyasset, so first inputs are (d,a,z,...)
% vfoptions.refine_d: only decisions d1,d3 are input to ReturnFn (and this model has no d1)

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
