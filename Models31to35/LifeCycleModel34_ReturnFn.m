function F=LifeCycleModel34_ReturnFn(savings,riskyshare,h,a,z,w,sigma,agej,Jr,pension,kappa_j,eta,psi)
% Note: using riskyasset, so first inputs are (d,a,z,...)
% riskyshare is not used here, but still has to be input as it is part of d

F=-Inf;
if agej<Jr % If working age
    c=w*kappa_j*z*h+a-savings; % Add h here
else % Retirement
    c=pension+a-savings;
end

if c>0
    F=(c^(1-sigma))/(1-sigma)+psi*((1-h)^(1-eta))/(1-eta); % The utility function
end

end
