function F=LifeCycleModel39B_ReturnFn(h,aprime,a,z,w,sigma,agej,Jr,pension,r,kappa_j,psi,eta)
% Now with endogenous labor

F=-Inf;
if agej<Jr % If working age
    c=w*kappa_j*z*h+(1+r)*a-aprime;
else % Retirement
    c=pension+(1+r)*a-z-aprime; % Subtract z here
end

if c>0
    F=(c^(1-sigma))/(1-sigma) -psi*(h^(1+eta))/(1+eta); % The utility function
end

end
