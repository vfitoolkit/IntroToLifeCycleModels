function F=LifeCycleModel40_ReturnFn(h,a1prime,a2prime,a1,a2,z1,z2,w,sigma,psi,eta,agej,Jr,pension,r1,r2,adjcost,kappa_j,wg1,wg2,wg3,beta,sj)
% As usual inputs are: decision variable, next period endogenous states
% (here two), this period endogenous states (here two), exogenous states (here two)
% After that we need all the parameters the return function uses, it
% doesn't matter what order we put them here.

F=-Inf;
if agej<Jr % If working age
    % a2 is an illiquid asset, so there is an adjustment cost
    adjcostincurred=0;
    if abs(a2-a2prime)>0.0001
        adjcostincurred=adjcost;
    end
    % budget constraint
    c=w*kappa_j*z1*z2*h+(1+r1)*a1+(1+r2)*a2-a1prime-a2prime-adjcostincurred;
else % Retirement
    % a2 is an illiquid asset, so there is an adjustment cost
    adjcostincurred=0;
    if abs(a2-a2prime)>0.0001
        adjcostincurred=adjcost;
    end
    % budget constraint
    c=pension++(1+r1)*a1+(1+r2)*a2-a1prime-a2prime-adjcostincurred;
end

if c>0
    F=(c^(1-sigma))/(1-sigma) -psi*(h^(1+eta))/(1+eta); % The utility function
end

% add the warm glow to the return, but only near end of life
if agej>=Jr+10
    % Warm glow of bequests: bequest are a luxury good
    warmglow=wg1*((1+(a1prime+a2prime)/wg2)^(1-wg3))/(1-wg3);
    % Modify for beta and sj (get the warm glow next period if die)
    warmglow=beta*(1-sj)*warmglow;
    % add the warm glow to the return
    F=F+warmglow;
end

end
