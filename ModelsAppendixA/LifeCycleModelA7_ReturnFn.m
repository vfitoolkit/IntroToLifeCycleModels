function F=LifeCycleModelA7_ReturnFn(h,aprime,a,z1,z2,z3,e1,e2,e3,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj)
% We first use the 'model space', which here is (d,aprime,a,z,e,...)
% After this everything is a parameter and VFI Toolkit will look for these
% in the parameter structure
%
% Six exogenous states (...,z1,z2,z3,e1,e2,e3,...)


F=-Inf;
if agej<Jr % If working age
    c=w*kappa_j*z1*z2*z3*e1*e2*e3*h+(1+r)*a-aprime;
else % Retirement
    c=pension+(1+r)*a-aprime;
end

if c>0
    F=(c^(1-sigma))/(1-sigma) -psi*(h^(1+eta))/(1+eta); % The utility function
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
