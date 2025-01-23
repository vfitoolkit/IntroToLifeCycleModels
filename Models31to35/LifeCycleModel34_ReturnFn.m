function F=LifeCycleModel34_ReturnFn(h,savings,a,z,w,sigma,agej,Jr,pension,kappa_j,eta,psi)
% Note: using riskyasset, so first inputs are (d,a,z,...)
% vfoptions.refine_d: only decisions d1,d3 are input to ReturnFn

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
