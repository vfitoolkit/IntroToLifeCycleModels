function F=LifeCycleModel43_ReturnFn(h,aprime,a,ebar,z,w,sigma,psi,eta,agej,Jr,r,kappa_j,pensionbp1,pensionbp2,pensionrate1,pensionrate2,pensionrate3,pensionscale,pensionmin,wg1,wg2,wg3,beta,sj)
% As usual the inputs are: decision variable, next period endogenous states,
% this period endogenous states, exogenous states.
% Here we have 1 decision variable, h, and two endogenous states, a and ebar.
% But ebar is an experienceassetz, so ebarprime is not chosen and so is not an
% input. Hence we have (h,aprime,a,ebar,z,...)
% After that we need all the parameters the return function uses, it
% doesn't matter what order we put them here.

F=-Inf;
if agej<Jr % If working age
    c=w*kappa_j*z*h+(1+r)*a-aprime;
else % Retirement
    % The pension is a piecewise-linear function of lifetime average
    % earnings ebar, with two bend points, and a minimum pension.
    pension=pensionscale*(pensionrate1*min(ebar,pensionbp1)+pensionrate2*max(min(ebar,pensionbp2)-pensionbp1,0)+pensionrate3*max(ebar-pensionbp2,0));
    pension=max(pension,pensionmin);
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
