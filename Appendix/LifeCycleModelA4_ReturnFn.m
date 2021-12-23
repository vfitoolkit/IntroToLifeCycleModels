function F=LifeCycleModelA4_ReturnFn(h,aprime,a,z1,z2,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj)
% z is AR(2), so z1 is current value and z2 is the lag
% While both z1 and z2 must be included because they are exogenous states 
% of the model we will only actually use the value of z1.

F=-Inf;
if agej<Jr % If working age
    c=w*kappa_j*z1*h+(1+r)*a-aprime; % Add z here
else % Retirement
    c=pension+(1+r)*a-aprime;
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
