function F=LifeCycleModel7_ReturnFn(h,aprime,a,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j,wg1,wg2,wg3,beta,sj)
% Modification is that we add the bottom half of this script that relates
% to the warm-glow of bequests. Note how we use 'agej>=Jr+10' to control
% which ages a warm-glow of bequests is received at. Also note that because
% the warm-glow occurs 'next period' we discount it with the discount
% factor beta (and 1-sj is the probability of dying).

F=-Inf;
if agej<Jr % If working age
    c=w*kappa_j*h+(1+r)*a-aprime;
else % Retirement
    c=pension+(1+r)*a-aprime;
end

if c>0
    F=(c^(1-sigma))/(1-sigma) -psi*(h^(1+eta))/(1+eta); % The utility function
end

% add the warm glow to the return, but only near end of life
if agej>=Jr+10
    % Warm glow of bequests: bequest are a luxury good
    warmglow=wg1*((1+aprime/wg2)^(1-wg3))/(1-wg3);
    % Modify for beta and sj (get the warm glow next period if die)
    warmglow=beta*(1-sj)*warmglow;
    % add the warm glow to the return
    F=F+warmglow;
end

end
