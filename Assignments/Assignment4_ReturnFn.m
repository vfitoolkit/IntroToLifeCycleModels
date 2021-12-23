function F=Assignment4_ReturnFn(aprime,a,z,w,sigma,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj,g)

F=-Inf;
if agej<Jr % If working age
    c=w*kappa_j*z+(1+r)*a-(1+g)*aprime; % Change to (1+g)aprime
else % Retirement
    c=pension+(1+r)*a-(1+g)*aprime; % Change to (1+g)aprime
end

if c>0
    F=(c^(1-sigma))/(1-sigma); % The utility function
end

% add the warm glow to the return, but only near end of life
if agej>=Jr+10
    % Warm glow of bequests
    warmglow=warmglow1*((((1+g)^agej)*aprime-warmglow2)^(1-warmglow3))/(1-warmglow3);
    % Modify for beta and sj (get the warm glow next period if die)
    warmglow=beta*(1-sj)*warmglow;
    % add the warm glow to the return
    F=F+warmglow;
end

end
