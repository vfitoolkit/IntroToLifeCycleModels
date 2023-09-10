function F=LifeCycleModel39_ReturnFn(aprime,a,z,w,sigma,agej,Jr,pension,r,kappa_j)
% Is LifeCycleModel21_ReturnFn, but modified for exogenous labor and to remove the warm-glow of bequests

F=-Inf;
if agej<Jr % If working age
    c=w*kappa_j*z+(1+r)*a-aprime;
else % Retirement
    c=pension+(1+r)*a-z-aprime; % Subtract z here
end

if c>0
    F=(c^(1-sigma))/(1-sigma); % The utility function
end

end
