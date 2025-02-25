function F=LifeCycleModel3_ReturnFn(h,aprime,a,w,sigma,psi,eta,agej,Jr,pension,r)
% In the baseline setup for VFI Toolkit, the first entries are always
% (i) decision variables, (ii) next period endogenous states, (iii) this 
% period endogenous states, and (iv) exogenous states.
% In this model we have 1 decision variable, h, 1 next period endogenous
% state, aprime, 1 this period endogenous state, a, and 0 exogenous states.
% Hence, we have (h,aprime,a,...)
% After that we need all the parameters the return function uses, it
% doesn't matter what order we put them here.

F=-Inf;
if agej<Jr % If working age
    c=w*h+(1+r)*a-aprime; % add assets, and subtract 'savings'
else % Retirement
    c=pension+(1+r)*a-aprime; % add assets, and subtract 'savings'
end

if c>0
    F=(c^(1-sigma))/(1-sigma) -psi*(h^(1+eta))/(1+eta); % The utility function
end

end
