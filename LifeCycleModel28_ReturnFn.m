function F=LifeCycleModel28_ReturnFn(h1,h2,aprime,a,z1,z2,e1,e2,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j_1,kappa_j_2,warmglow1,warmglow2,warmglow3,beta,sj)
% The first four are the 'always required' decision variables, next period
% endogenous states, this period endogenous states, exogenous states
% After that we need all the parameters the return function uses, it
% doesn't matter what order we put them here.

F=-Inf;
if agej<Jr % If working age
    c=w*kappa_j_1*z1*e1*h1+w*kappa_j_2*z2*e2*h2+(1+r)*a-aprime; % Add z here
else % Retirement
    c=pension+(1+r)*a-aprime;
end

if c>0
    F=(c^(1-sigma))/(1-sigma) -psi*(h1^(1+eta))/(1+eta)-psi*(h2^(1+eta))/(1+eta); % The utility function
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
