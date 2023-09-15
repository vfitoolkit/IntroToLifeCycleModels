function F=LifeCycleModel1_ReturnFn(h,aprime,a,w,sigma,psi,eta)
% The first four are the 'always required' decision variables, next period
% endogenous states, this period endogenous states, exogenous states
% After that we need all the parameters the return function uses, it
% doesn't matter what order we put them here.

F=-Inf; % -Inf is used as 'never do this'; it is only used if not overwritten below

c=w*h; % This is the budget constraint

% We need to check that consumption is positive, otherwise utility is -Inf
% (Note that this is already the value of F and will be returned if we
% don't satisfy c>0)
if c>0
    F=(c^(1-sigma))/(1-sigma) -psi*(h^(1+eta))/(1+eta); % The utility function
end

end
