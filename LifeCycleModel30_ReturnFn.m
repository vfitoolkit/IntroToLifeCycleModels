function F=LifeCycleModel30_ReturnFn(h,f,aprime,a,n1,n2,z,w,sigma,psi,eta,agej,eta1,eta2,eta3,nbar,hbar,h_c,childcarec,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj)

F=-Inf;

infanttime=h_c*n1; % Time cost of infants
childcarecosts=childcarec*n1*(h>0); % Cost of childcare for infants if working


if agej<Jr % If working age
    c=w*kappa_j*z*h+(1+r)*a-childcarecosts-aprime;
else % Retirement
    c=pension+(1+r)*a-aprime;
end

leisure=hbar-h-infanttime;

consumption_equiv_units=1+0.3*n1+0.5*n2; % I just made this scale up as a placeholder

utility_of_children=eta1*exp(agej-eta3)/(1+exp(agej-eta3)) *(nbar+n1+n2)^eta2;

if c>0 && leisure<1
    F=((c/consumption_equiv_units)^(1-sigma))/(1-sigma)-psi*((hbar-leisure)^(1+eta2))/(1+eta) +utility_of_children;
    % F=log(c/consumption_equiv_units)+psi*log(leisure)+utility_of_children;
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

% if f>0
%     F=F-1;
% end


end
