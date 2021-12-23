function F=LifeCycleModel5_ReturnFn(h,aprime,a,z,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j)
% The first four are the 'always required' decision variables, next period
% endogenous states, this period endogenous states, exogenous states
% After that we need all the parameters the return function uses, it
% doesn't matter what order we put them here.

F=-Inf;
if agej<Jr % If working age
    c=w*kappa_j*h+(1+r)*a-aprime; % add assets, and subtract 'savings'
else % Retirement
    c=pension+(1+r)*a-aprime; % add assets, and subtract 'savings'
end

if c>0
    F=(c^(1-sigma))/(1-sigma) -psi*(h^(1+eta))/(1+eta); % The utility function
end

end
