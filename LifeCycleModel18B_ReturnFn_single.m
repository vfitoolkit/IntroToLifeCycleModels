function F=LifeCycleModel18B_ReturnFn_single(aprime,a,z,w,sigma,agej,Jr,pension,r,kappa_j,wg1,wg2,wg3,beta,sj,meanearningsratio)
% The first four are the 'always required' decision variables, next period
% endogenous states, this period endogenous states, exogenous states
% After that we need all the parameters the return function uses, it
% doesn't matter what order we put them here.

F=single(-Inf);
single_1=single(1);
if agej<Jr % If working age
    c=meanearningsratio*w*kappa_j*z+(single_1+r)*a-aprime; % z is stochastic endowment
else % Retirement
    c=pension+(single_1+r)*a-aprime;
end

if c>0
    F=(c^(single_1-sigma))/(single_1-sigma); % The utility function
end

% add the warm glow to the return, but only near end of life
if agej>=Jr+10
    % Warm glow of bequests: bequest are a luxury good
    warmglow=wg1*((single_1+aprime/wg2)^(single_1-wg3))/(single_1-wg3);
    % Modify for beta and sj (get the warm glow next period if die)
    warmglow=beta*(single_1-sj)*warmglow;
    % add the warm glow to the return
    F=F+warmglow;
end

end
