function F=LifeCycleModel8_ReturnFn_single(h,aprime,a,z,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j,wg1,wg2,wg3,beta,sj)
% In the baseline setup for VFI Toolkit, the first entries are always
% (i) decision variables, (ii) next period endogenous states, (iii) this 
% period endogenous states, and (iv) exogenous states.
% In this model we have 1 decision variable, h, 1 next period endogenous
% state, aprime, 1 this period endogenous state, a, and 1 markov exogenous state, z.
% Hence, we have (h,aprime,a,z,...)
% After that we need all the parameters the return function uses, it
% doesn't matter what order we put them here.

% Important change: we now have z as the fourth input to the ReturnFn, the
% space of our model has increased.

F=single(-Inf);
single_1=single(1);

if agej<Jr % If working age
    c=w*kappa_j*z*h+(single_1+r)*a-aprime; % Add z here
else % Retirement
    c=pension+(single_1+r)*a-aprime;
end

if c>0
    F=(c^(single_1-sigma))/(single_1-sigma) -psi*(h^(single_1+eta))/(single_1+eta); % The utility function
end

% add the warm glow to the return, but only near end of life
if agej>=Jr+single(10)
    % Warm glow of bequests: bequest are a luxury good
    warmglow=wg1*((single_1+aprime/wg2)^(single_1-wg3))/(single_1-wg3);
    % Modify for beta and sj (get the warm glow next period if die)
    warmglow=beta*(single_1-sj)*warmglow;
    % add the warm glow to the return
    F=F+warmglow;
end

end
