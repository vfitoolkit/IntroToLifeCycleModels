function F=LifeCycleModel41_ReturnFn(p,aprime,a,h,z,w,sigma,psi,y_m,childcarecosts,agej,Jr,pension,r,warmglow1,warmglow2,warmglow3,beta,sj)

F=-Inf;
if agej<Jr % If working age
    c=w*h*z*p-childcarecosts*p+y_m+(1+r)*a-aprime; % Add z here
else % Retirement
    c=pension+(1+r)*a-aprime;
end

if c>0
    F=(c^(1-sigma))/(1-sigma) -psi*p; % The utility function
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
