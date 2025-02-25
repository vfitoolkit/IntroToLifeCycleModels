function F=LifeCycleModel10_ReturnFn(aprime,a,z,w,sigma,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj)
% In the baseline setup for VFI Toolkit, the first entries are always
% (i) decision variables, (ii) next period endogenous states, (iii) this 
% period endogenous states, and (iv) exogenous states.
% In this model we have 0 decision variables, 1 next period endogenous
% state, aprime, 1 this period endogenous state, a, and 1 markov exogenous state, z.
% Hence, we have (aprime,a,z,...)
% After that we need all the parameters the return function uses, it
% doesn't matter what order we put them here.

F=-Inf;
if agej<Jr % If working age
    c=w*kappa_j*z+(1+r)*a-aprime; % z is stochastic endowment
else % Retirement
    c=pension+(1+r)*a-aprime;
end

if c>0
    F=(c^(1-sigma))/(1-sigma); % The utility function
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
