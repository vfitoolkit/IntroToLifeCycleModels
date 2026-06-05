function F=LifeCycleModel1_ReturnFn_single(h,aprime,a,w,sigma,psi,eta)
% In the baseline setup for VFI Toolkit, the first entries are always
% (i) decision variables, (ii) next period endogenous states, (iii) this 
% period endogenous states, and (iv) exogenous states.
% In this model we have 1 decision variable, h, 1 next period endogenous
% state, aprime, 1 this period endogenous state, a, and 0 exogenous states.
% Hence, we have (h,aprime,a,...)
% After that we need all the parameters the return function uses, it
% doesn't matter what order we put them here.

F=single(-Inf); % -Inf is used as 'never do this'; it is only used if not overwritten below

c=w*h; % This is the budget constraint

% We need to check that consumption is positive, otherwise utility is -Inf
% (Note that this is already the value of F and will be returned if we
% don't satisfy c>0)
if c>0
    single_1=single(1);
    F=(c^(single_1-sigma))/(single_1-sigma) -psi*(h^(single_1+eta))/(single_1+eta); % The utility function
end

end
