function F=Assignment2_ReturnFn(h,aprime,a,z,w,sigma1,sigma2,sigma3,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj)
% The first four are the 'always required' decision variables, next period
% endogenous states, this period endogenous states, exogenous states
% After that we need all the parameters the return function uses, it
% doesn't matter what order we put them here.

F=-Inf;
if agej<Jr % If working age
    c=w*kappa_j*z*h+(1+r)*a-aprime; % Add z here
else % Retirement
    c=pension+(1+r)*a-aprime;
end

if c>0
    F=((sigma1*c^sigma2 +(1-sigma1)*((1-h)^(1-sigma2)))^(1-sigma3))/(1-sigma3); % Non-seperable utility function
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
