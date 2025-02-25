function F=LifeCycleModel21_ReturnFn(h,aprime,a,z,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj)
% Is LifeCycleModel8_ReturnFn, but modified to include medical expense
% shocks when retired.

F=-Inf;
if agej<Jr % If working age
    c=w*kappa_j*z*h+(1+r)*a-aprime;
else % Retirement
    c=pension+(1+r)*a-z-aprime; % Subtract z here
end

if c>0
    F=(c^(1-sigma))/(1-sigma) -psi*(h^(1+eta))/(1+eta); % The utility function
end

% add the warm glow to the return, but only near end of life
if agej>=Jr+10
    % Warm glow of bequests
    warmglow=warmglow1*((aprime-warmglow2)^(1-warmglow3))/(1-warmglow3);
    % Modify for beta and sj (get the warm glow next period if die)
    warmglow=beta*(1-sj)*warmglow;
    % add the warm glow to the return
    F=F+warmglow;
end

end
