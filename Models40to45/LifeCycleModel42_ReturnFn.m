function F=LifeCycleModel42_ReturnFn(l,s,aprime,a,h,z,w,sigma,eta,psi,agej,Jr,pension,r,warmglow1,warmglow2,warmglow3,beta,sj)

leisure=1-s-l; % one unit of time, minus time spent studying and working

F=-Inf;
if agej<Jr % If working age
    c=w*h*z*l+(1+r)*a-aprime; % Add z here
else % Retirement
    c=pension+(1+r)*a-aprime;
end

if c>0 && leisure>0
    F=(c^(1-sigma))/(1-sigma) +psi*(leisure^(1-eta))/(1-eta); % The utility function
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
